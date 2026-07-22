# Copyright 2024 Kyutai and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generation logic for the Moshi model."""

from dataclasses import dataclass
from typing import Any

import torch

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import ModelOutput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


def _grow_static_cache(past_key_values: Cache, target_length: int, sliding_window: int | None = None) -> None:
    """Widen a preallocated cache in place so a resumed generation can keep writing past its original end.

    A static cache is sized once, from the `max_length` of the call that created it. A stream resuming from that
    cache asks for more room than the first chunk ever needed, and a sliding-window layer answers by rolling its
    oldest entry out -- silently dropping context that is still inside the model's real window. Layers are grown
    to `target_length` (capped at the window they are actually meant to hold), keeping the entries already written.

    Growth is geometric, as for any growable buffer: sized to the exact request, every chunk of a stream would
    reallocate and copy the whole cache, which is quadratic over a conversation and runs into hundreds of
    megabytes per chunk once it is a few thousand frames long. Doubling amortises that to a constant per frame.

    A layer that has already rolled is left alone: its contents are a window offset from the start of the
    sequence, and the mask arithmetic reads that offset off `max_cache_len`, so widening it there would misplace
    every cached key. Growing before each chunk keeps a stream from ever reaching that point.
    """
    for layer in getattr(past_key_values, "layers", []):
        max_cache_len = getattr(layer, "max_cache_len", None)
        if max_cache_len is None or not getattr(layer, "is_initialized", False):
            continue
        # A sliding layer is capped at its window, past which rolling is the intended behaviour, not data loss.
        # The layer folds the window into `max_cache_len` at build time and keeps no record of it, so it is passed in.
        window = sliding_window if getattr(layer, "is_sliding", False) else None
        new_length = max(target_length, 2 * max_cache_len)
        new_length = new_length if window is None else min(new_length, window)
        if new_length <= max_cache_len or layer.get_seq_length() > max_cache_len:
            continue

        length = int(layer.get_seq_length())
        for name in ("keys", "values"):
            old = getattr(layer, name)
            new = old.new_zeros((*old.shape[:2], new_length, old.shape[3]))
            new[:, :, :length] = old[:, :, :length]
            setattr(layer, name, new)
            # The address the compiled graph was handed is gone; the new one has to be pinned in its place.
            torch._dynamo.mark_static_address(new)
        layer.max_cache_len = new_length


@dataclass
class MoshiStreamingState:
    """
    Everything one call to [`MoshiForConditionalGeneration.generate`] has to know about the calls before it.

    Moshi's loop carries more between steps than a text model's does: two audio histories, the delay-pattern masks
    that describe how those histories are laid out over time, and the hidden state the depth decoder turns into the
    closing frame. A caller streaming a conversation hands this object back and the next chunk picks up exactly
    where the last one stopped, instead of re-deriving all of it from a prompt -- which is what makes a chunked
    generation disagree with a single-shot one.

    It is also the only place that state lives. `generate` and the hooks it overrides thread this object through
    `model_kwargs` rather than stashing anything on the model, so one model can drive as many conversations as the
    caller has states for.

    Attributes:
        past_key_values (`Cache`, *optional*):
            The main decoder's KV cache.
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The text stream so far. Stands in for `input_ids` on the next chunk.
        assistant_audio_codes (`torch.Tensor` of shape `(batch_size, num_codebooks, sequence_length)`, *optional*):
            Moshi's own audio history, *undelayed*. The delay pattern is a wire format applied where the codes are
            read, not the stored form: writing it back here would shift the same codes a second time.
        user_audio_codes (`torch.Tensor` of shape `(batch_size, num_codebooks, sequence_length)`, *optional*):
            The user history the loop reads, held at the same length as `assistant_audio_codes` and read at the same
            index. Frames the caller supplied are forced onto it through `user_delay_pattern_mask`; the rest hold
            the depth decoder's prediction, or zeros where it predicts no user stream.
        supplied_user_audio_codes (`torch.Tensor` of shape `(batch_size, num_codebooks, sequence_length)`, *optional*):
            The user audio the caller has sent, accumulated across the stream. This is what the delay mask is
            rebuilt from when new audio arrives, which is why it is kept apart from `user_audio_codes`: only real
            input belongs in the mask, and a stream nobody is speaking into has to stay open for prediction.
        predicted_user_audio_codes (`torch.Tensor` of shape `(batch_size, num_codebooks, sequence_length)`, *optional*):
            The user stream the depth decoder invented, when it predicts both sides. Returned to the caller.
        user_delay_pattern_mask (`torch.Tensor`, *optional*):
            Delay pattern for the user stream, at absolute positions over the whole conversation.
        assistant_delay_pattern_mask (`torch.Tensor`, *optional*):
            Delay pattern for Moshi's own stream.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, 1, hidden_size)`, *optional*):
            The frame the stream stopped on, which the depth decoder needs to close it out.
        prompt_frames (`int`):
            Length of the prompt the stream opened on. Counted with the frame `concat_unconditional_inputs`
            prepends, so it lines up with `supplied_user_audio_codes`, and used to tell the caller's own audio
            apart from what the depth decoder filled in.
    """

    past_key_values: Cache | None = None
    sequences: torch.LongTensor | None = None
    assistant_audio_codes: torch.Tensor | None = None
    user_audio_codes: torch.Tensor | None = None
    supplied_user_audio_codes: torch.Tensor | None = None
    predicted_user_audio_codes: torch.Tensor | None = None
    user_delay_pattern_mask: torch.Tensor | None = None
    assistant_delay_pattern_mask: torch.Tensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    prompt_frames: int = 0

    @property
    def user_supplied_frames(self) -> int:
        """How many generated frames the caller supplied the user audio for, rather than the depth decoder."""
        supplied = 0 if self.supplied_user_audio_codes is None else self.supplied_user_audio_codes.shape[-1]
        return max(supplied - self.prompt_frames, 0)


@auto_docstring(
    custom_intro="""
    Outputs of [`MoshiForConditionalConditionalGeneration.generate`].
    """
)
@dataclass
class MoshiConditionalGenerationGenerateOutput(ModelOutput):
    r"""
    sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
        The generated text sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
        if all batches finished early due to the `eos_token_id`.
    sequences_scores (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True`):
        Final beam scores of the generated `sequences`.
    scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
        Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
        of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
        Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
        with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
    logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
        Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
        at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
        each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
    beam_indices (`torch.LongTensor`, *optional*, returned when `output_scores=True`):
        Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
        `(batch_size*num_return_sequences, sequence_length)`.
    attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
        Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
        `torch.FloatTensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
    hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
        Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
        `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
    past_key_values (`Cache`, *optional*, returned when `use_cache=True`):
        Contains the model cache, used to speed up decoding. Different models have a different cache format, check
        the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
    audio_codes (`torch.LongTensor` of shape `(batch_size*num_return_sequences, num_codeooks, sequence_length)`, *optional*):
        The generated audio codes. Returned if `return_audio_codes=True`. Turn them into a waveform with [`MoshiProcessor.decode_audio`].
    user_audio_codes (`torch.LongTensor` of shape `(batch_size*num_return_sequences, num_codeooks, sequence_length)`, *optional*):
        The user stream the depth decoder predicted for the frames the caller did not supply. Only present when the
        depth decoder predicts both streams and the caller left part of the horizon open.
    streaming_state ([`MoshiStreamingState`], *optional*):
        Everything a following chunk needs to carry on. Hand it back as `generate(..., streaming_state=...)`,
        together with whatever user audio has arrived since, and the next chunk continues the conversation instead
        of starting a new one from the prompt.
    """

    sequences: torch.LongTensor | None = None
    sequences_scores: torch.FloatTensor | None = None
    scores: tuple[torch.FloatTensor] | None = None
    logits: tuple[torch.FloatTensor] | None = None
    beam_indices: torch.LongTensor | None = None
    attentions: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[tuple[torch.FloatTensor]] | None = None
    past_key_values: Cache | None = None
    audio_codes: torch.LongTensor | None = None
    user_audio_codes: torch.LongTensor | None = None
    streaming_state: MoshiStreamingState | None = None


@auto_docstring
@dataclass
class MoshiUnconditionalInput(ModelOutput):
    r"""
    input_ids (`torch.Tensor `of shape `(batch_size, sequence_length), *optional*):
        The sequence used as a text prompt for the generation.
    user_audio_codes (`torch.Tensor `of shape `(batch_size, num_codebooks, sequence_length), *optional*):
        The audio codes used as audio user prompt for the generation, as produced by [`MoshiProcessor`].
    assistant_audio_codes (`torch.Tensor `of shape `(batch_size, num_codebooks, sequence_length), *optional*):
        The audio codes used as audio Moshi prompt for the generation, as produced by [`MoshiProcessor`].
    attention_mask (`torch.LongTensor`)  of shape `(batch_size, sequence_length)`, *optional*):
        Attention mask to avoid performing attention on padding token indices. Mask values selected in `[0,
        1]`: 1 for tokens that are **not masked**, 0 for tokens that are **masked**.
    """

    input_ids: torch.LongTensor | None = None
    user_audio_codes: torch.Tensor | None = None
    assistant_audio_codes: torch.Tensor | None = None
    attention_mask: torch.LongTensor | None = None


class MoshiGenerationMixin(GenerationMixin):
    """
    Generation loop for [`MoshiForConditionalGeneration`].

    Moshi interleaves two decoders: at every step the main decoder produces a hidden state, the depth decoder
    turns it into one audio token per codebook, and those tokens are re-embedded together with the text token to
    form the next step's input. That does not fit `GenerationMixin`'s default loop, so the hooks below are
    overridden. The delay-pattern helpers live here as well since they describe how codebooks are laid out over
    time for generation (`forward` reuses `build_delay_pattern_mask` through the MRO).

    Everything the loop carries between steps lives on a [`MoshiStreamingState`], threaded through `model_kwargs`
    so that the hooks can reach it. Nothing is kept on the model: a `generate` call leaves it as it found it, and
    a caller can drive several conversations through the same weights at once.
    """

    def _split_predicted_streams(self, predicted_codes: torch.Tensor):
        """
        Split what the depth decoder produced into the assistant stream and, when it predicts both, the user one.

        Codebooks are laid out one stream after the other, the assistant's first, matching how `forward` feeds them
        in (`cat([assistant_audio_codes, user_audio_codes], dim=1)`) and how upstream indexes them (agent at
        `1 + q`, user at `AUDIO_TOKENS_PER_STREAM + 1 + q`).
        """
        assistant_codes = predicted_codes[:, : self.num_codebooks]
        user_codes = predicted_codes[:, self.num_codebooks :] if self.predicts_user_stream else None
        return assistant_codes, user_codes

    def _embed_audio_codes(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """
        Sum the per-codebook audio embeddings.

        `audio_codes` spans both streams, the assistant's codebooks followed by the user's, matching the layout of
        `embed_tokens`. That is a `ModuleList`, so under a device map its entries can land on different devices;
        each embedding's output is therefore moved to a single device before summing.
        """
        target_device = self.embed_tokens[0].weight.device
        return sum(
            self.embed_tokens[codebook](audio_codes[:, codebook]).to(target_device)
            for codebook in range(audio_codes.shape[1])
        )

    def _prepare_inputs_embeds_for_generation(
        self,
        input_ids: torch.LongTensor | None,
        user_audio_codes: torch.Tensor,
        assistant_audio_codes: torch.Tensor,
    ) -> torch.FloatTensor:
        """
        Embed one step's worth of the three streams into the vector the decoder consumes.

        Both audio streams have to arrive already carrying the delay pattern, and both are needed: `embed_tokens`
        holds Moshi's codebooks followed by the user's, so embedding one alone silently drops the other's
        contribution. The prompt and the generation loop both come through here, with the same shapes.
        """
        # The two streams can reach here from different places (caller inputs vs `get_unconditional_inputs`), and
        # under a device map those are not necessarily the same device. Line them up with the embeddings that are
        # about to consume them.
        codes_device = self.embed_tokens[0].weight.device
        audio_codes = torch.cat([assistant_audio_codes.to(codes_device), user_audio_codes.to(codes_device)], dim=1)
        inputs_embeds = self._embed_audio_codes(audio_codes)
        if input_ids is not None:
            inputs_embeds = inputs_embeds + self.model.embed_tokens(input_ids).to(inputs_embeds.device)
        return inputs_embeds

    def _concat_unconditional_inputs(self, input_ids, user_audio_codes, assistant_audio_codes, attention_mask):
        """
        Open the conversation on a frame that says nobody has spoken yet, prepended to every stream at once.

        Every absolute position downstream -- the delay masks, the audio histories, the KV cache -- is measured
        from here, which is why a resumed chunk must not add it a second time and why `prompt_frames` counts it.
        """
        unconditional_inputs = self.get_unconditional_inputs(num_samples=user_audio_codes.shape[0])
        assistant_audio_codes = torch.cat(
            [unconditional_inputs.assistant_audio_codes.to(assistant_audio_codes), assistant_audio_codes], dim=2
        )
        user_audio_codes = torch.cat(
            [unconditional_inputs.user_audio_codes.to(user_audio_codes), user_audio_codes], dim=2
        )
        input_ids = torch.cat([unconditional_inputs.input_ids.to(input_ids), input_ids], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat([unconditional_inputs.attention_mask.to(attention_mask), attention_mask], dim=1)
        return input_ids, user_audio_codes, assistant_audio_codes, attention_mask

    def _prepare_delay_masks(self, streaming_state, user_audio_codes, assistant_audio_codes, horizon):
        """
        Build the delay-pattern masks for this call, or stretch the ones the stream is already using.

        A mask is read at absolute positions over the whole conversation, so it has to reach the end of this
        chunk's horizon. The two streams stretch differently. Moshi's own is opened up with `-1` -- "predict here".
        The user's cannot be: `-1` tells the loop to fall back to a prediction, which would silently drop audio
        that has just arrived, so the caller's new frames are written into it instead, exactly as a cold call
        handed the whole stream up front would have written them.
        """
        audio_vocab_size = self.config.audio_vocab_size
        if streaming_state is None:
            _, assistant_mask = self.build_delay_pattern_mask(
                assistant_audio_codes, bos_token_id=audio_vocab_size, pad_token_id=audio_vocab_size, max_length=horizon
            )
            _, user_mask = self.build_delay_pattern_mask(
                user_audio_codes, bos_token_id=audio_vocab_size, pad_token_id=audio_vocab_size, max_length=horizon
            )
            return user_mask, assistant_mask

        assistant_mask = streaming_state.assistant_delay_pattern_mask
        if assistant_mask is not None:
            assistant_mask = self.extend_delay_pattern_mask(assistant_mask, horizon)

        user_mask = streaming_state.user_delay_pattern_mask
        supplied = streaming_state.supplied_user_audio_codes
        if user_mask is not None and horizon > user_mask.shape[-1]:
            if supplied is None:
                user_mask = self.extend_delay_pattern_mask(user_mask, horizon)
            else:
                # Only the frames past the old horizon are written, which is all `build_delay_pattern_mask` would
                # have produced differently: the delay is a fixed one-frame shift for every codebook but the first,
                # so the layout of what came before does not depend on what arrives later. Rebuilding the whole
                # mask and copying the old head back over it -- which is what this used to do -- costs the length
                # of the conversation on every chunk, for a tail that is a chunk long.
                #
                # The frames already decided keep the layout they went through, all but one: the slot the previous
                # horizon's pad was sitting on holds real audio now, since the stream did not end there.
                decided = min(user_mask.shape[-1] - 1, horizon)
                supplied_frames = min(supplied.shape[-1], horizon)
                extended = user_mask.new_full((*user_mask.shape[:-1], horizon), -1)
                extended[..., :decided] = user_mask[..., :decided]
                # The first codebook is not shifted, the rest read a frame back. Past the audio that has arrived
                # the mask stays `-1`, i.e. open for the depth decoder to speak for the user.
                if supplied_frames > decided:
                    extended[:, 0, decided:supplied_frames] = supplied[:, 0, decided:supplied_frames]
                shift_start, shift_end = max(decided, 1), min(supplied_frames + 1, horizon)
                if shift_end > shift_start:
                    extended[:, 1:, shift_start:shift_end] = supplied[:, 1:, shift_start - 1 : shift_end - 1]
                user_mask = extended
        return user_mask, assistant_mask

    def _prepare_prompt_embeds(
        self,
        input_ids,
        user_audio_codes,
        assistant_audio_codes,
        user_delay_pattern_mask,
        assistant_delay_pattern_mask,
        past_key_values,
    ):
        """
        Embed the prompt, computing only the frames the model has not already read.

        A resumed chunk hands the whole conversation over, most of which is in the KV cache: the base loop takes the
        tail with `inputs_embeds[:, -next_sequence_length:]` and never looks at the rest. But it also reads
        `next_sequence_length` off `inputs_embeds.shape[1]`, so the tensor has to be full length even though its
        head is dead weight. Embedding that head costs one gather per codebook per frame of the conversation, on
        every chunk; zeros stand in for it instead.
        """
        length = input_ids.shape[-1] if input_ids is not None else assistant_audio_codes.shape[-1]
        past_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        past_length = int(past_length) if torch.is_tensor(past_length) else past_length
        read_from = min(past_length, length - 1) if past_length else 0

        embeds = self._prepare_inputs_embeds_for_generation(
            input_ids if input_ids is None else input_ids[:, read_from:],
            user_audio_codes=self.apply_delay_pattern_mask(
                user_audio_codes[..., read_from:], user_delay_pattern_mask[..., read_from:]
            ),
            assistant_audio_codes=self.apply_delay_pattern_mask(
                assistant_audio_codes[..., read_from:], assistant_delay_pattern_mask[..., read_from:]
            ),
        )
        if read_from == 0:
            return embeds

        prompt_embeds = embeds.new_zeros((embeds.shape[0], length, embeds.shape[-1]))
        prompt_embeds[:, read_from:] = embeds
        return prompt_embeds

    def _open_streaming_state(self, input_ids, user_audio_codes, assistant_audio_codes, num_sequences):
        """
        Start a stream from a prompt: seed both histories with it and record what the caller supplied.

        Both histories go in *undelayed*, matching how the loop extends them. Reading them back off a delayed
        tensor is not equivalent -- the delay shifts every codebook but the first, so a prompt that went through
        it once and came back would be shifted again, one frame per round trip.

        The user history only ever takes the prompt, even when the caller handed over a stream that runs past it:
        the loop appends one frame per step, so seeding it with the whole horizon would run past the delay mask.
        The frames beyond the prompt are forced through that mask instead, which is what makes the caller's audio
        win over the depth decoder's prediction.
        """
        prompt_frames = assistant_audio_codes.shape[-1]
        user_history = user_audio_codes[:, :, :prompt_frames]
        return MoshiStreamingState(
            sequences=input_ids,
            assistant_audio_codes=torch.repeat_interleave(assistant_audio_codes, num_sequences, dim=0),
            user_audio_codes=torch.repeat_interleave(user_history, num_sequences, dim=0),
            prompt_frames=prompt_frames,
        )

    def _close_streaming_state(self, streaming_state, output_text_ids, kwargs_depth_decoder):
        """
        Turn the frame the stream stopped on into audio.

        The loop produces a hidden state for the last text token but never gets to spend it: the depth decoder runs
        at the *start* of a step, on the step before's hidden state. One more pass here gives that last frame its
        audio, which is also what the next chunk reads back as the frame before its own first step.
        """
        last_hidden_state = streaming_state.last_hidden_state
        last_hidden_state = last_hidden_state.view(-1, 1, last_hidden_state.shape[-1])

        frame = self.depth_decoder.generate(
            last_hidden_state=last_hidden_state,
            input_ids=output_text_ids[:, -1:].view(-1, 1),
            **kwargs_depth_decoder,
        )
        # Drop the leading text token, then split the frame into the two streams.
        assistant_frame, user_frame = self._split_predicted_streams(frame[:, 1:].unsqueeze(2))

        streaming_state.assistant_audio_codes = torch.cat(
            [streaming_state.assistant_audio_codes, assistant_frame], dim=2
        )
        # The user history has to grow with it. The loop keeps the two the same length and reads them at the same
        # index, so a user stream left a frame short here lags one frame behind the assistant one for the whole of
        # a resumed stream. Where the depth decoder predicts no user stream the placeholder is zeros, exactly as in
        # the loop -- `user_delay_pattern_mask` writes the caller's own frame over it wherever there is one.
        if user_frame is None:
            user_frame = streaming_state.user_audio_codes.new_zeros((*streaming_state.user_audio_codes.shape[:2], 1))
        else:
            # The returned prediction gets this frame too, for the same reason the assistant stream does: the two
            # are decoded side by side, and a user stream a frame short of the assistant one drifts out of sync --
            # by a whole frame per chunk once a caller is streaming.
            streaming_state.predicted_user_audio_codes = (
                user_frame
                if streaming_state.predicted_user_audio_codes is None
                else torch.cat([streaming_state.predicted_user_audio_codes, user_frame], dim=2)
            )
        streaming_state.user_audio_codes = torch.cat([streaming_state.user_audio_codes, user_frame], dim=2)

    def _streaming_audio_outputs(self, streaming_state):
        """Trim the two histories down to what the caller asked for."""
        # The history is already what the caller wants: the depth decoder emits a whole frame at a time, and the
        # delay is applied only where the codes are fed back in. Applying it here and slicing it off again would
        # not cancel out -- the slice shifts every codebook but the first, so codes that were never delayed come
        # out misaligned by a frame, which is audible as slurred speech and makes a step-by-step caller's output
        # disagree with the same codes decoded in one go.
        audio_codes = streaming_state.assistant_audio_codes

        # Frames the caller did not supply an assistant stream for hold the audio BOS id -- "Moshi has not spoken
        # yet". Mimi has no such code, so returning them would hand the codec an index past its codebooks. They
        # only ever sit at the front, and no real code can collide with the reserved id, so the leading run of
        # them is trimmed.
        is_bos_frame = (audio_codes == self.config.audio_vocab_size).all(dim=1).all(dim=0)
        if is_bos_frame.any():
            spoken = (~is_bos_frame).nonzero()
            first_spoken = int(spoken[0]) if spoken.numel() else audio_codes.shape[-1]
            audio_codes = audio_codes[:, :, first_spoken:]

        # Only meaningful when the depth decoder predicted the user stream for frames the caller left open.
        user_audio_codes = None
        if streaming_state.predicted_user_audio_codes is not None:
            predicted = streaming_state.predicted_user_audio_codes[:, :, streaming_state.user_supplied_frames :]
            user_audio_codes = predicted if predicted.shape[-1] > 0 else None
        return audio_codes, user_audio_codes

    def generate(
        self,
        input_ids: torch.LongTensor | None = None,
        user_audio_codes: torch.Tensor | None = None,
        assistant_audio_codes: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        return_audio_codes: bool | None = True,
        concat_unconditional_inputs: bool | None = True,
        streaming_state: MoshiStreamingState | None = None,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Generates sequences of text token ids and audio tokens ids.

        Parameters:
            input_ids (`torch.Tensor `of shape `(batch_size, sequence_length), *optional*):
                The sequence used as a text prompt for the generation. Taken from `streaming_state` when resuming.
            user_audio_codes (`torch.Tensor `of shape `(batch_size, num_codebooks, sequence_length), *optional*):
                The audio codes used as audio user prompt for the generation, as produced by [`MoshiProcessor`].
                When resuming, only the frames that have arrived since the last call: the stream so far is on
                `streaming_state` and this is appended to it.
            assistant_audio_codes (`torch.Tensor `of shape `(batch_size, num_codebooks, sequence_length), *optional*):
                The audio codes used as audio Moshi prompt for the generation, as produced by [`MoshiProcessor`].
                Taken from `streaming_state` when resuming.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` and the audio inputs you can choose to directly pass an embedded representation. This
                is useful if you want more control over how to convert the inputs into associated vectors than the
                model's internal embedding lookup matrix.
            return_audio_codes (`bool`, *optional*, defaults to `True`):
                If `True`, will also returns the generated audio codes, i.e the intermediate audio "tokens" which transforms to `audio_sequences` once passed through the audio decoder.
            concat_unconditional_inputs (`bool`, *optional*, defaults to `True`):
                If `False`, won't concatenate initial audio and text tokens. Ignored when resuming: the stream
                already opened on that frame and every absolute position is measured from it.
            streaming_state ([`MoshiStreamingState`], *optional*):
                The state a previous call returned. Hand it back, with whatever user audio has arrived since, to
                carry that conversation on rather than starting a new one:

                ```python
                out = model.generate(**inputs, max_new_tokens=frames)
                out = model.generate(
                    user_audio_codes=new_frames, max_new_tokens=frames, streaming_state=out.streaming_state
                )
                ```
            kwargs (`dict[str, Any]`, *optional*):
                Remaining dictionary of keyword arguments that are passed to the `generate` method. Refers to the
                original [`generate` docstrings](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate)
                for more information on how to use them.
                Note that keywords with a *depth_* prefix will be input for the `generate` method of the
                depth decoder. Otherwise, the latter will use its default generation config.
        Return:
            [`MoshiConditionalGenerationGenerateOutput`]
        """
        # multiple generate -> need to create/update device map
        if hasattr(self, "hf_device_map") and not hasattr(self.depth_decoder, "hf_device_map"):
            self.depth_decoder.hf_device_map = {}
            if "" in self.hf_device_map:
                self.depth_decoder.hf_device_map = self.hf_device_map
            else:
                main_device = [d for d in self.hf_device_map.values() if d not in ["cpu", "disk"]][0]
                self.depth_decoder.hf_device_map = {
                    key[len("depth_decoder") :]: main_device if value in ["cpu", "disk"] else value
                    for key, value in self.hf_device_map.items()
                    if key.startswith("depth_decoder")
                }
            # need to remove depth_decoder from the top device_map so that we assign correctly the device for each layer idx in the cache
            self.hf_device_map = {
                key: value for key, value in self.hf_device_map.items() if not key.startswith("depth_decoder")
            }
        # retrieve depth decoder kwargs
        depth_decoder_kwargs_keys = {argument for argument in kwargs if argument.startswith("depth_decoder_")}
        kwargs_depth_decoder = {
            argument[len("depth_decoder_") :]: kwargs.pop(argument) for argument in depth_decoder_kwargs_keys
        }

        # needs to prepare generation config, even though it'll be done again in `generate`
        generation_config, kwargs = self._prepare_generation_config(kwargs.pop("generation_config", None), **kwargs)

        # Resuming: the state is the conversation, so a streaming caller passes only the audio that has just
        # arrived. Anything it does pass on top still wins, which is what lets a caller drive the model its own way.
        if streaming_state is not None:
            concat_unconditional_inputs = False
            kwargs.setdefault("past_key_values", streaming_state.past_key_values)
            if input_ids is None and inputs_embeds is None:
                input_ids = streaming_state.sequences
            if assistant_audio_codes is None:
                assistant_audio_codes = streaming_state.assistant_audio_codes
            # Accumulated for the delay mask, which is rebuilt at absolute positions over the whole conversation.
            carried = streaming_state.supplied_user_audio_codes
            if user_audio_codes is not None:
                carried = (
                    user_audio_codes if carried is None else torch.cat([carried, user_audio_codes.to(carried)], dim=2)
                )
            streaming_state.supplied_user_audio_codes = carried
            # What goes downstream has to cover the conversation so far, and the caller's own audio need not: a
            # caller who stops sending leaves the depth decoder to speak for the user, and what it said is on the
            # loop's history. The two are laid over each other, the caller's audio winning where it reaches -- the
            # same order the delay mask imposes anyway, so this only decides what the checks below are handed.
            history = streaming_state.user_audio_codes
            if carried is None:
                user_audio_codes = history
            elif carried.shape[-1] >= history.shape[-1]:
                user_audio_codes = carried
            else:
                user_audio_codes = torch.cat([carried, history[..., carried.shape[-1] :]], dim=2)

        attention_mask = kwargs.pop("attention_mask", None)
        # Whether there is a user on the other end has to be read before the streams are filled in below: a side
        # that never speaks is handed a placeholder, and forcing that onto the delay mask would tell the loop the
        # user said something rather than leaving the stream open for the depth decoder to predict.
        caller_supplied_user_audio = user_audio_codes is not None

        input_ids, user_audio_codes, assistant_audio_codes, concat_unconditional_inputs = (
            self._check_and_maybe_initialize_inputs(
                input_ids=input_ids,
                user_audio_codes=user_audio_codes,
                assistant_audio_codes=assistant_audio_codes,
                inputs_embeds=inputs_embeds,
                concat_unconditional_inputs=concat_unconditional_inputs,
                # One frame, i.e. a prompt saying "the user has not spoken yet". Stretching that id across the
                # whole horizon would instead claim the user never speaks at all, which is not what silence
                # sounds like to Moshi: the codec encodes silence to ordinary codes, and feeding the reserved id
                # in their place is off-distribution enough to turn the generated text into word salad. A caller
                # who wants to generate without a live user passes real silence from
                # `MoshiProcessor.get_silence_audio_codes`; the check below points them there.
                num_user_frames=1,
            )
        )

        if attention_mask is None and input_ids is not None:
            # Not derived by comparing `input_ids` against `pad_token_id`, the way the base implementation does.
            # Moshi's text stream starts on `vocab_size`, and the released checkpoints set `pad_token_id` to that
            # same id, so that comparison marks the prompt itself as padding and the model attends to nothing --
            # silently, and the text stream degenerates. Every frame handed to `generate` is real content here.
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        if concat_unconditional_inputs:
            input_ids, user_audio_codes, assistant_audio_codes, attention_mask = self._concat_unconditional_inputs(
                input_ids, user_audio_codes, assistant_audio_codes, attention_mask
            )

        inputs = inputs_embeds if input_ids is None else input_ids
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name="inputs_embeds" if input_ids is None else "input_ids",
            inputs_tensor=inputs,
            input_ids_length=inputs.shape[-1],
        )
        # Kept because `max_length` is handed back unresolved below, and both the masks and the cache are sized
        # from the horizon it resolved to.
        resolved_max_length = generation_config.max_length

        # Moshi is full-duplex and consumes a user frame at every step. A depth decoder that predicts the user
        # stream can fill in whatever the caller did not supply; one that only predicts Moshi's own stream cannot,
        # so there the stream has to reach the end of the horizon -- `MoshiProcessor.get_silence_audio_codes`
        # produces silence for the whole span when there is no live user.
        if (
            not self.predicts_user_stream
            and user_audio_codes is not None
            and user_audio_codes.shape[-1] < resolved_max_length
        ):
            raise ValueError(
                f"`user_audio_codes` covers {user_audio_codes.shape[-1]} frames but generation runs to "
                f"{resolved_max_length}. Moshi consumes a user frame at every step, so the stream has to "
                "reach the end of the horizon. To generate without a live user, hand it encoded silence rather "
                "than a shorter stream:\n"
                "    inputs = model.get_unconditional_inputs()\n"
                f"    inputs.user_audio_codes = processor.get_silence_audio_codes({resolved_max_length})\n"
                "    model.generate(**inputs, max_new_tokens=...)"
            )

        # retrieve depth decoder generation config if it exists
        if hasattr(generation_config, "depth_decoder_config"):
            depth_decoder_generation_config = generation_config.depth_decoder_config
        else:
            # we need to control the number of tokens generated by the depth decoder
            # One token per codebook the depth decoder predicts (`dep_q`), plus the leading text token. This is
            # the depth decoder's own count, which is larger than the parent's per-stream one when the user-side
            # heads are kept.
            num_depth_codebooks = self.depth_decoder.config.num_codebooks
            depth_decoder_generation_config = {
                "min_length": num_depth_codebooks + 1,
                "max_length": num_depth_codebooks + 1,
                "cache_implementation": "static",
            }
        # update kwargs_depth_decoder: kwargs_depth_decoder have priority over depth_decoder_generation_config
        depth_decoder_generation_config.update(kwargs_depth_decoder)
        kwargs_depth_decoder = depth_decoder_generation_config

        # The masks reach at least the end of this chunk, and at least as far as the user audio that came with it.
        horizon = resolved_max_length
        if streaming_state is not None and streaming_state.supplied_user_audio_codes is not None:
            horizon = max(horizon, streaming_state.supplied_user_audio_codes.shape[-1])
        user_delay_pattern_mask, assistant_delay_pattern_mask = self._prepare_delay_masks(
            streaming_state, user_audio_codes, assistant_audio_codes, horizon
        )
        # A caller driving the delay pattern itself still wins.
        caller_user_mask = kwargs.pop("user_delay_pattern_mask", None)
        caller_assistant_mask = kwargs.pop("assistant_delay_pattern_mask", None)
        if caller_user_mask is not None:
            user_delay_pattern_mask = caller_user_mask
        if caller_assistant_mask is not None:
            assistant_delay_pattern_mask = caller_assistant_mask

        # The prompt goes in through the same two masks the loop reads at every step, so a resumed chunk re-reads
        # the frames it already decided exactly as it decided them. On a resume both streams come off the state --
        # a self-playing checkpoint has nothing else to read the user side from, since the caller never sent one.
        # They stay at the batch size the caller passed: beams and return sequences are expanded by the base loop,
        # and only the histories the loop keeps growing are seeded expanded.
        if streaming_state is None:
            prompt_user_audio_codes = user_audio_codes[:, :, : assistant_audio_codes.shape[-1]]
            prompt_assistant_audio_codes = assistant_audio_codes
        else:
            prompt_user_audio_codes = streaming_state.user_audio_codes
            prompt_assistant_audio_codes = streaming_state.assistant_audio_codes

        if streaming_state is None:
            num_sequences = max(generation_config.num_beams, generation_config.num_return_sequences)
            streaming_state = self._open_streaming_state(
                input_ids, user_audio_codes, assistant_audio_codes, num_sequences
            )
            # In mask coordinates, i.e. counting the frame prepended above, so it lines up with `prompt_frames`.
            streaming_state.supplied_user_audio_codes = user_audio_codes if caller_supplied_user_audio else None
        streaming_state.user_delay_pattern_mask = user_delay_pattern_mask
        streaming_state.assistant_delay_pattern_mask = assistant_delay_pattern_mask

        if inputs_embeds is None:
            inputs_embeds = self._prepare_prompt_embeds(
                input_ids,
                prompt_user_audio_codes,
                prompt_assistant_audio_codes,
                user_delay_pattern_mask,
                assistant_delay_pattern_mask,
                kwargs.get("past_key_values"),
            )

        # The masks had to be built out to the horizon, which is why `max_length` was resolved early. The base loop
        # resolves it again from `max_new_tokens` and reads a `max_length` that is already set as one the caller
        # asked for, so it warns about a clash with itself on every call. Hand it back the unresolved config: it
        # recomputes the same horizon, since by now `input_ids` carries the frame that was prepended above.
        generation_config.max_length = None

        # Beam search needs the scores and the beam indices to reorder the audio history afterwards, so both are
        # forced on. They go on the config rather than alongside it: passing generation arguments next to a
        # `generation_config` is deprecated, and doing it here warned on every call.
        # What the caller actually asked for, kept because the checks below distinguish "the caller wanted a dict"
        # from "this call needed one anyway".
        caller_wants_dict = generation_config.return_dict_in_generate
        # Always on: the KV cache reaches the streaming state through the base loop's dict output, and a state
        # handed back without one still resumes -- by re-reading the whole conversation into a fresh cache, which
        # is both quadratic and, because the mask arithmetic is written for a cache that carries over, not quite
        # the same audio. A caller who does not ask for a dict gets `sequences` below exactly as before.
        generation_config.return_dict_in_generate = True
        generation_config.output_scores = generation_config.num_beams > 1 or generation_config.output_scores

        # A caller resuming a stream hands back the KV cache from the previous chunk. `generation_config` asks for
        # a fresh `cache_implementation`, which the base loop refuses to combine with a supplied cache. Clearing it
        # on `generation_config` alone is not enough: the base re-fills any `None` field from
        # `self.generation_config`, so the model default is cleared for the call and restored right after.
        resuming = kwargs.get("past_key_values") is not None
        saved_cache_implementation = self.generation_config.cache_implementation
        if resuming:
            generation_config.cache_implementation = None
            self.generation_config.cache_implementation = None
            # That cache was sized for the chunk that built it, which is shorter than the stream it is now carrying.
            _grow_static_cache(
                kwargs["past_key_values"], resolved_max_length, getattr(self.config, "sliding_window", None)
            )
        try:
            outputs = super().generate(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                generation_config=generation_config,
                kwargs_depth_decoder=kwargs_depth_decoder,
                attention_mask=attention_mask,
                streaming_state=streaming_state,
                user_delay_pattern_mask=user_delay_pattern_mask,
                assistant_delay_pattern_mask=assistant_delay_pattern_mask,
                **kwargs,
            )
        finally:
            if resuming:
                self.generation_config.cache_implementation = saved_cache_implementation

        streaming_state.past_key_values = getattr(outputs, "past_key_values", None)
        streaming_state.sequences = outputs.sequences

        if not return_audio_codes:
            return outputs if caller_wants_dict else outputs.sequences

        if generation_config.num_beams > 1:
            self._reorder_streaming_state_for_beams(
                streaming_state, outputs.beam_indices, assistant_audio_codes, generation_config
            )

        self._close_streaming_state(streaming_state, outputs.sequences, kwargs_depth_decoder)
        audio_codes, predicted_user_audio_codes = self._streaming_audio_outputs(streaming_state)

        if caller_wants_dict:
            return MoshiConditionalGenerationGenerateOutput(
                audio_codes=audio_codes,
                user_audio_codes=predicted_user_audio_codes,
                streaming_state=streaming_state,
                **outputs,
            )

        return MoshiConditionalGenerationGenerateOutput(
            sequences=outputs.sequences,
            audio_codes=audio_codes,
            user_audio_codes=predicted_user_audio_codes,
            streaming_state=streaming_state,
        )

    def _reorder_streaming_state_for_beams(
        self, streaming_state, beam_indices, assistant_audio_codes, generation_config
    ):
        """Rebuild the audio history and the last hidden state along the beams that actually survived."""
        # Beam indices are of shape `input_length + number_generated_tokens` but actually starts
        # indexing indices at index 0 instead of index `input_length-1`.
        # We thus discard the last `input_length` indices that are never used.
        prompt_length = assistant_audio_codes.shape[-1]
        beam_indices = beam_indices[:, :-prompt_length]

        generated_audio_codes = streaming_state.assistant_audio_codes[:, :, prompt_length:]
        # we've generated audio tokens `number_generated_tokens-1` times, so we use the corresponding beam indices
        # to retrieve the right audio tokens
        expanded_beam_indices = beam_indices[:, :-1].unsqueeze(1).expand(-1, self.num_codebooks, -1)
        generated_audio_codes = torch.gather(generated_audio_codes, dim=0, index=expanded_beam_indices)

        assistant_audio_codes = torch.repeat_interleave(
            assistant_audio_codes, generation_config.num_return_sequences, dim=0
        )
        streaming_state.assistant_audio_codes = torch.cat((assistant_audio_codes, generated_audio_codes), dim=2)
        streaming_state.last_hidden_state = torch.index_select(
            streaming_state.last_hidden_state, dim=0, index=beam_indices[:, -1]
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        logits_to_keep=None,
        streaming_state=None,
        user_delay_pattern_mask=None,
        assistant_delay_pattern_mask=None,
        kwargs_depth_decoder=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- Moshi has custom post-processing on the prepared inputs.

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            user_delay_pattern_mask=user_delay_pattern_mask,
            assistant_delay_pattern_mask=assistant_delay_pattern_mask,
            kwargs_depth_decoder=kwargs_depth_decoder,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # 2. Now that everything is prepared, generate audio_codes using the depth decoder

        # we want to do it after a first token has been generated
        if model_inputs["input_ids"] is not None:
            last_hidden_state = kwargs.pop("last_hidden_state")
            # (batch_size, sequence_length, dim) -> (batch_size * sequence_length, 1, dim)
            last_hidden_state = last_hidden_state.view(-1, 1, last_hidden_state.shape[-1])

            # One step per row from here on: the depth decoder and the embeddings below both read a single frame,
            # so a step axis is folded into the batch rather than carried alongside it. The loop only ever gets
            # here with one token per row, which makes this a no-op there.
            text_ids = model_inputs.pop("input_ids").view(-1, 1)

            frame = self.depth_decoder.generate(
                last_hidden_state=last_hidden_state,
                input_ids=text_ids,
                **kwargs_depth_decoder,
            )

            # the first tokens are text tokens
            assistant_frame, predicted_user_frame = self._split_predicted_streams(frame[:, 1:].unsqueeze(2))

            # Advance the user stream by one frame. `user_delay_pattern_mask` carries whatever the caller supplied
            # and is `-1` past the end of it, so only the unsupplied frames take the tensor concatenated here --
            # the caller always wins, as upstream's per-slot `provided` guard does. Those frames get the depth
            # decoder's own prediction when it predicts the user stream, and zeros otherwise (in which case
            # `generate` has already refused a stream that falls short of the horizon).
            user_frame = predicted_user_frame
            if user_frame is None:
                user_frame = streaming_state.user_audio_codes.new_zeros(
                    (*streaming_state.user_audio_codes.shape[:2], 1)
                )
            streaming_state.user_audio_codes = torch.cat([streaming_state.user_audio_codes, user_frame], dim=2)
            user_audio_codes = self.apply_delay_pattern_mask(
                streaming_state.user_audio_codes, user_delay_pattern_mask
            )[:, :, -1:]
            if predicted_user_frame is not None:
                # The *undelayed* prediction, for the same reason the assistant history is kept undelayed: the
                # delayed frame above is a wire format for feeding the model back, and returning it would hand the
                # caller codes whose every codebook but the first is a frame out of place.
                streaming_state.predicted_user_audio_codes = (
                    predicted_user_frame
                    if streaming_state.predicted_user_audio_codes is None
                    else torch.cat([streaming_state.predicted_user_audio_codes, predicted_user_frame], dim=2)
                )
            # Kept undelayed. The delay pattern is a wire format, not the history itself: writing it back into the
            # state would mean a later `build_delay_pattern_mask` -- which shifts whatever it is handed -- shifts
            # the same codes a second time. It is applied where the codes are read instead.
            streaming_state.assistant_audio_codes = torch.cat(
                [streaming_state.assistant_audio_codes, assistant_frame], dim=2
            )
            delayed_audio_codes = self.apply_delay_pattern_mask(
                streaming_state.assistant_audio_codes, assistant_delay_pattern_mask
            )

            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = self._prepare_inputs_embeds_for_generation(
                text_ids,
                user_audio_codes=user_audio_codes,
                assistant_audio_codes=delayed_audio_codes[:, :, -1:],
            )

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )

        # update last_hidden_state that'll be used in the depth decoder. ``.clone()`` breaks the
        # view into the main decoder's cudagraph output buffer — otherwise the depth decoder reads
        # the slice *after* the next main-decoder step has already overwritten it.
        last_hidden_state = outputs.get("last_hidden_state")[:, -1:].clone()
        model_kwargs["last_hidden_state"] = last_hidden_state
        # The loop reads it from `model_kwargs` on the next step; the state carries it past the end of the loop,
        # where the closing depth decoder pass turns the last frame into audio.
        model_kwargs["streaming_state"].last_hidden_state = last_hidden_state
        return model_kwargs

    @staticmethod
    def extend_delay_pattern_mask(decoder_pad_token_mask, max_length: int, pad_token_id: int | None = None):
        """
        Stretch a delay-pattern mask to a longer horizon, for a stream that has run past the one it was built for.

        A mask is built once for a fixed `max_length` (see `build_delay_pattern_mask`), which a chunked caller does
        not know up front: each chunk pushes the horizon out. Only two entries in it are horizon-dependent -- the
        pad closing the first codebook, and the length itself. The rest is either a prompt value or `-1`, meaning
        "keep whatever is passed", which is exactly right for the frames that have not arrived yet.

        The pad is lifted and, unless `pad_token_id` says otherwise, not written again: it marks the end of a
        generation, and a stream that is being extended has not reached one. Left in place it would force a pad onto
        a slot that is about to hold real audio.
        """
        current_length = decoder_pad_token_mask.shape[-1]
        if current_length >= max_length:
            return decoder_pad_token_mask

        extended = decoder_pad_token_mask.new_full((*decoder_pad_token_mask.shape[:-1], max_length), -1)
        extended[..., :current_length] = decoder_pad_token_mask
        extended[:, 0, current_length - 1] = -1
        if pad_token_id is not None:
            extended[:, 0, -1] = pad_token_id
        return extended

    @staticmethod
    def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask):
        """Apply a delay pattern mask to the decoder input ids, only preserving predictions where
        the mask is set to -1, and otherwise setting to the value detailed in the mask."""
        seq_len = input_ids.shape[-1]
        decoder_pad_token_mask = decoder_pad_token_mask[..., :seq_len]
        input_ids = torch.where(decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask)
        return input_ids

    def build_delay_pattern_mask(
        self, input_ids: torch.LongTensor, bos_token_id: int, pad_token_id: int, max_length: int | None = None
    ):
        """Build a delayed pattern mask to the input_ids. Each codebook, except the first one, is offset by
        one, giving a delayed pattern mask at the start of sequence and end of sequence. Take the example where there
        are 4 codebooks and a max sequence length of 6, we have the delayed pattern mask of shape `(codebooks,
        seq_len)`:
        - [-1, -1, -1, -1, -1,  P]
        - [ B, -1, -1, -1, -1, -1]
        - [ B, -1, -1, -1, -1, -1]
        - [ B, -1, -1, -1, -1, -1]
        where B is the beginning-of-sentence token, P is the special padding token id and -1 indicates that the token is valid for prediction. If we include
        a prompt (input ids), the -1 positions indicate where new tokens should be predicted. Otherwise, the
        mask is set to the value in the prompt:
        - [ a0, a1, -1, -1, -1,  P]
        - [ B,  b0, b1, -1, -1, -1]
        - [ B,  c0, c1, -1, -1, -1]
        - [ B,  d0, d1, -1, -1, -1]
        where a-d indicate the codebook channel and 0/1 indicates the temporality. Now, we only override the -1
        tokens in our prediction.

        Returns a `(shifted_input_ids, pattern_mask)` pair: the codes are *shifted* into the delayed layout, and
        the mask records the forced positions. Calling this again on codes that already went through it shifts
        them a second time, so it belongs at the start of a generation only. Once the mask exists, later frames go
        through `apply_delay_pattern_mask`, which writes the forced values without shifting -- that is what the
        generation loop does at every step, and what a caller driving `generate` a frame at a time should pass the
        mask back for.
        """
        bsz, num_codebooks, seq_len = input_ids.shape

        max_length = max_length if max_length is not None else self.generation_config.max_length
        input_ids_shifted = (
            torch.ones((bsz, num_codebooks, max_length), dtype=torch.long, device=input_ids.device) * -1
        )

        # the first codebook channel is not shifted
        seq_len_to_keep = min(seq_len, max_length - 1)
        input_ids_shifted[:, 0, :seq_len_to_keep] = input_ids[:, 0, :seq_len_to_keep]

        # fill the shifted ids with the prompt entries
        input_ids_shifted[:, 1:, 1 : seq_len_to_keep + 1] = input_ids[:, 1:, :seq_len_to_keep]

        # fill with BOS and PAD
        input_ids_shifted[:, 1:, 0] = bos_token_id
        input_ids_shifted[:, 0, -1] = pad_token_id

        # construct a pattern mask that indicates the positions of BOS and PAD tokens for each codebook
        pattern_mask = input_ids_shifted

        input_ids = input_ids_shifted[..., :seq_len_to_keep]
        return input_ids, pattern_mask

    def get_unconditional_inputs(self, num_samples=1, num_user_frames=1):
        """
        Helper function to get null inputs for unconditional generation, enabling the model to be used without the
        feature extractor or tokenizer.

        Args:
            num_samples (int, *optional*):
                Number of audio samples to unconditionally generate.
            num_user_frames (int, *optional*):
                Length of the returned user stream. It has to reach the end of the generation horizon.
            max_new_tokens (int, *optional*):
                Number of tokens to generate for each sample. More tokens means longer audio samples, at the expense of
                longer inference (since more audio tokens need to be generated per sample).

        Example:
        ```python
        >>> from transformers import MoshiForConditionalGeneration

        >>> model = MoshiForConditionalGeneration.from_pretrained("kmhf/hf-moshiko-pytorch-bf16")

        >>> # get the unconditional (or 'null') inputs for the model
        >>> unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
        >>> audio_samples = model.generate(**unconditional_inputs, max_new_tokens=256)
        ```"""

        input_ids = torch.ones((num_samples, 1), device=self.device, dtype=torch.int64) * self.config.vocab_size
        # Moshi consumes a user frame at every generated step, so callers that will generate need the stream to
        # reach the end of the horizon: `num_user_frames` is `1 + max_new_tokens` in that case.
        user_audio_codes = (
            torch.ones((num_samples, self.num_codebooks, num_user_frames), device=self.device, dtype=torch.int64)
            * self.config.audio_vocab_size
        )
        assistant_audio_codes = (
            torch.ones((num_samples, self.num_codebooks, 1), device=self.device, dtype=torch.int64)
            * self.config.audio_vocab_size
        )
        attention_mask = torch.ones((num_samples, 1), device=self.device, dtype=torch.long)

        return MoshiUnconditionalInput(
            input_ids=input_ids,
            user_audio_codes=user_audio_codes,
            assistant_audio_codes=assistant_audio_codes,
            attention_mask=attention_mask,
        )

    def _check_and_maybe_initialize_inputs(
        self,
        input_ids=None,
        user_audio_codes=None,
        assistant_audio_codes=None,
        inputs_embeds=None,
        concat_unconditional_inputs=None,
        num_user_frames=1,
    ):
        inputs = input_ids if inputs_embeds is None else inputs_embeds
        user_input = user_audio_codes
        assistant_input = assistant_audio_codes

        one_input_has_been_passed = (user_input is not None) or (assistant_input is not None) or (inputs is not None)

        # concat_unconditional_inputs will be False if inputs_embeds is used
        concat_unconditional_inputs = concat_unconditional_inputs and not (
            inputs_embeds is not None and input_ids is None
        )

        # A stream the caller left out is filled with the audio BOS id, the same way `get_unconditional_inputs`
        # does: it means "this side has not spoken yet". That is the normal shape of a first turn -- the processor
        # only produces `assistant_audio_codes` when it is handed `assistant_audio` -- so the missing streams are
        # filled in rather than rejected, taking their length from whichever stream the caller did pass.
        if one_input_has_been_passed:
            reference = next(x for x in (inputs, user_input, assistant_input) if x is not None)
            batch_size, prompt_length, device = reference.shape[0], reference.shape[-1], reference.device

            def _silent_stream():
                return torch.full(
                    (batch_size, self.num_codebooks, prompt_length),
                    self.config.audio_vocab_size,
                    device=device,
                    dtype=torch.int64,
                )

            if user_audio_codes is None:
                user_audio_codes = user_input = _silent_stream()
            if assistant_audio_codes is None:
                assistant_audio_codes = assistant_input = _silent_stream()
            if inputs is None:
                # The text stream opens on the same reserved id the audio streams do.
                input_ids = inputs = torch.full(
                    (batch_size, prompt_length), self.config.vocab_size, device=device, dtype=torch.int64
                )

        if one_input_has_been_passed and (user_input is None):
            raise ValueError(
                "No user audio inputs have been passed alongside the other inputs. Make sure `user_audio_codes` is passed or use `MoshiForConditionalGeneration.get_unconditional_inputs`. Check the `MoshiForConditionalGeneration` docstrings for more information."
            )
        elif one_input_has_been_passed and (assistant_input is None):
            raise ValueError(
                "No Moshi audio inputs have been passed alongside the other inputs. Make sure `assistant_audio_codes` is passed or use `MoshiForConditionalGeneration.get_unconditional_inputs`. Check the `MoshiForConditionalGeneration` docstrings for more information."
            )
        elif one_input_has_been_passed and (inputs is None):
            raise ValueError(
                "No `input_ids` or `inputs_embeds` have been passed alongside the other inputs. Make sure `input_ids` is passed or use `MoshiForConditionalGeneration.get_unconditional_inputs`. Check the `MoshiForConditionalGeneration` docstrings for more information."
            )
        elif not one_input_has_been_passed:
            # if no inputs have been passed, use default values
            # Nothing was passed, so there is no user either: hand generation a user stream that reaches the end
            # of the horizon.
            unconditional_inputs = self.get_unconditional_inputs(num_user_frames=num_user_frames)
            input_ids = unconditional_inputs.input_ids
            user_audio_codes = unconditional_inputs.user_audio_codes
            assistant_audio_codes = unconditional_inputs.assistant_audio_codes

            # in that case, no need to concat unconditional inputs
            concat_unconditional_inputs = False
        else:
            # check if same sequence length
            user_seq_length = user_input.shape[-1]
            assistant_seq_length = assistant_input.shape[-1]
            tokens_seq_length = inputs.shape[1]

            # The text and assistant streams describe what has already happened, so they must line up exactly. The
            # user stream may run ahead: Moshi consumes a user frame at every step, so a caller who already knows
            # what the user says can hand the whole thing over instead of feeding it one frame at a time.
            if tokens_seq_length != assistant_seq_length or user_seq_length < tokens_seq_length:
                raise ValueError(
                    f"`input_ids` ({tokens_seq_length}) and `assistant_audio_codes` ({assistant_seq_length}) must have "
                    f"the same sequence length, and `user_audio_codes` ({user_seq_length}) must be at least as long. "
                    "Check the `MoshiForConditionalGeneration` docstrings for more information."
                )

        return input_ids, user_audio_codes, assistant_audio_codes, concat_unconditional_inputs
