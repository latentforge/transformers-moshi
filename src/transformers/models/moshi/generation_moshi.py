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
from ...generation import GenerationConfig, GenerationMixin
from ...modeling_outputs import ModelOutput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


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
        input_ids: torch.LongTensor | None = None,
        user_audio_codes: torch.Tensor | None = None,
        assistant_audio_codes: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        apply_delay_pattern_mask: bool = False,
        concat_unconditional_inputs: bool = False,
        user_delay_pattern_mask: torch.Tensor | None = None,
        assistant_delay_pattern_mask: torch.Tensor | None = None,
    ):

        if inputs_embeds is None and input_ids is None and user_audio_codes is None and assistant_audio_codes is None:
            raise ValueError(
                "You must provide at least one of `input_ids`, `user_audio_codes`, `assistant_audio_codes` or `inputs_embeds`."
            )

        if inputs_embeds is None and concat_unconditional_inputs:
            unconditional_inputs = self.get_unconditional_inputs(num_samples=user_audio_codes.shape[0])
            assistant_audio_codes = torch.cat(
                [unconditional_inputs.assistant_audio_codes, assistant_audio_codes], dim=2
            )
            user_audio_codes = torch.cat([unconditional_inputs.user_audio_codes, user_audio_codes], dim=2)
            input_ids = torch.cat([unconditional_inputs.input_ids, input_ids], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([unconditional_inputs.attention_mask, attention_mask], dim=1)

        # Moshi's own stream before the delay is written into it. The generation loop keeps its history in this
        # undelayed form -- the delay is a wire format applied where the codes are read -- so it has to start from
        # the undelayed prompt too. Reading it back off the delayed tensor below is not equivalent: the delay
        # shifts every codebook but the first, so a prompt that went through it once and is handed back as the
        # next call's prompt would be shifted again, one frame per round trip.
        undelayed_assistant_audio_codes = assistant_audio_codes

        if inputs_embeds is None or apply_delay_pattern_mask:
            # A caller driving `generate` a step at a time hands back the mask it got, so the delay is *applied*
            # here rather than rebuilt. `build_delay_pattern_mask` shifts whatever it is given, so rebuilding it
            # from an already-advanced history would shift those codes a second time; `apply_delay_pattern_mask`
            # only writes the mask's forced values, which is what the generation loop does at every step.
            if apply_delay_pattern_mask and user_audio_codes is not None and user_delay_pattern_mask is not None:
                user_audio_codes = self.apply_delay_pattern_mask(user_audio_codes, user_delay_pattern_mask)
            elif apply_delay_pattern_mask and user_audio_codes is not None:
                user_audio_codes, user_delay_pattern_mask = self.build_delay_pattern_mask(
                    user_audio_codes,
                    bos_token_id=self.config.audio_vocab_size,
                    pad_token_id=self.config.audio_vocab_size,
                    max_length=generation_config.max_length,
                )

            if (
                apply_delay_pattern_mask
                and assistant_audio_codes is not None
                and assistant_delay_pattern_mask is not None
            ):
                assistant_audio_codes = self.apply_delay_pattern_mask(
                    assistant_audio_codes, assistant_delay_pattern_mask
                )
            elif apply_delay_pattern_mask and assistant_audio_codes is not None:
                assistant_audio_codes, assistant_delay_pattern_mask = self.build_delay_pattern_mask(
                    assistant_audio_codes,
                    bos_token_id=self.config.audio_vocab_size,
                    pad_token_id=self.config.audio_vocab_size,
                    max_length=generation_config.max_length,
                )

        # If inputs_embeds is provided, it has the priority over input_ids and audio_codes, which won't be used
        if inputs_embeds is None:
            # The user stream may run ahead of the prompt (the caller can hand over future frames up front). Only
            # the part that lines up with the text and assistant streams belongs in the prompt embeddings; the rest
            # is consumed one frame at a time by the generation loop, through `user_delay_pattern_mask`.
            if user_audio_codes is not None and assistant_audio_codes is not None:
                prompt_length = assistant_audio_codes.shape[-1]
                if user_audio_codes.shape[-1] > prompt_length:
                    user_audio_codes = user_audio_codes[..., :prompt_length]
            # The two streams can reach here from different places (caller inputs vs `get_unconditional_inputs`),
            # and under a device map those are not necessarily the same device. Line them up with the embeddings
            # that are about to consume them.
            codes_device = self.embed_tokens[0].weight.device
            if user_audio_codes is not None:
                user_audio_codes = user_audio_codes.to(codes_device)
            if assistant_audio_codes is not None:
                assistant_audio_codes = assistant_audio_codes.to(codes_device)

            # Moshi always embeds both streams together: `embed_tokens` holds the assistant's codebooks followed by
            # the user's, and every caller reaches here with both (`_check_and_maybe_initialize_inputs` fills in
            # whichever the caller left out). Embedding one alone would silently drop the other's contribution.
            audio_inputs_embeds = None
            if user_audio_codes is not None and assistant_audio_codes is not None:
                audio_codes = torch.cat([assistant_audio_codes, user_audio_codes], dim=1)
                audio_inputs_embeds = self._embed_audio_codes(audio_codes)
            elif user_audio_codes is not None or assistant_audio_codes is not None:
                missing = "assistant_audio_codes" if user_audio_codes is not None else "user_audio_codes"
                raise ValueError(
                    f"Both audio streams are needed to build the inputs embeddings, but `{missing}` is missing."
                )

            if input_ids is not None:
                inputs_embeds = self.model.embed_tokens(input_ids)

            if audio_inputs_embeds is not None:
                inputs_embeds = (
                    audio_inputs_embeds
                    if inputs_embeds is None
                    else audio_inputs_embeds + inputs_embeds.to(audio_inputs_embeds.device)
                )

        return (
            inputs_embeds,
            input_ids,
            user_audio_codes,
            assistant_audio_codes,
            undelayed_assistant_audio_codes,
            user_delay_pattern_mask,
            assistant_delay_pattern_mask,
            attention_mask,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor | None = None,
        user_audio_codes: torch.Tensor | None = None,
        assistant_audio_codes: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        return_audio_codes: bool | None = True,
        concat_unconditional_inputs: bool | None = True,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Generates sequences of text token ids and audio tokens ids.

        Parameters:
            input_ids (`torch.Tensor `of shape `(batch_size, sequence_length), *optional*):
                The sequence used as a text prompt for the generation.
            user_audio_codes (`torch.Tensor `of shape `(batch_size, num_codebooks, sequence_length), *optional*):
                The audio codes used as audio user prompt for the generation, as produced by [`MoshiProcessor`].
            assistant_audio_codes (`torch.Tensor `of shape `(batch_size, num_codebooks, sequence_length), *optional*):
                The audio codes used as audio Moshi prompt for the generation, as produced by [`MoshiProcessor`].
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` and the audio inputs you can choose to directly pass an embedded representation. This
                is useful if you want more control over how to convert the inputs into associated vectors than the
                model's internal embedding lookup matrix.
            return_audio_codes (`bool`, *optional*, defaults to `True`):
                If `True`, will also returns the generated audio codes, i.e the intermediate audio "tokens" which transforms to `audio_sequences` once passed through the audio decoder.
            concat_unconditional_inputs (`bool`, *optional*, defaults to `True`):
                If `False`, won't concatenate initial audio and text tokens.
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

        # The loop advances the user stream one frame per step, starting right after the prompt. Whatever the
        # caller supplied beyond the prompt covers the first few of those, so only the frames past it are the depth
        # decoder's own predictions -- the rest is just the caller's input handed back.
        prompt_frames = assistant_audio_codes.shape[-1] if assistant_audio_codes is not None else 0
        supplied_frames = user_audio_codes.shape[-1] if user_audio_codes is not None else 0
        self._user_supplied_steps = max(supplied_frames - prompt_frames, 0)

        inputs = inputs_embeds if input_ids is None else input_ids

        input_ids_length = inputs.shape[-1] + 1 if concat_unconditional_inputs else inputs.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name="inputs_embeds" if input_ids is None else "input_ids",
            inputs_tensor=inputs,
            input_ids_length=input_ids_length,
        )

        # Moshi is full-duplex and consumes a user frame at every step. A depth decoder that predicts the user
        # stream can fill in whatever the caller did not supply; one that only predicts Moshi's own stream cannot,
        # so there the stream has to reach the end of the horizon -- `MoshiProcessor.get_silence_audio_codes`
        # produces silence for the whole span when there is no live user.
        if (
            not self.predicts_user_stream
            and user_audio_codes is not None
            and user_audio_codes.shape[-1] < generation_config.max_length
        ):
            raise ValueError(
                f"`user_audio_codes` covers {user_audio_codes.shape[-1]} frames but generation runs to "
                f"{generation_config.max_length}. Moshi consumes a user frame at every step, so the stream has to "
                "reach the end of the horizon. To generate without a live user, hand it encoded silence rather "
                "than a shorter stream:\n"
                "    inputs = model.get_unconditional_inputs()\n"
                f"    inputs.user_audio_codes = processor.get_silence_audio_codes({generation_config.max_length})\n"
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

        attention_mask = kwargs.pop("attention_mask", None)
        if attention_mask is None:
            # Not derived by comparing `input_ids` against `pad_token_id`, the way the base implementation does.
            # Moshi's text stream starts on `vocab_size`, and the released checkpoints set `pad_token_id` to that
            # same id, so that comparison marks the prompt itself as padding and the model attends to nothing --
            # silently, and the text stream degenerates. Every frame handed to `generate` is real content here.
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        (
            inputs_embeds,
            input_ids,
            user_audio_codes,
            assistant_audio_codes,
            undelayed_assistant_audio_codes,
            user_delay_pattern_mask,
            assistant_delay_pattern_mask,
            attention_mask,
        ) = self._prepare_inputs_embeds_for_generation(
            input_ids=input_ids,
            user_audio_codes=user_audio_codes,
            assistant_audio_codes=assistant_audio_codes,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            apply_delay_pattern_mask=True,
            concat_unconditional_inputs=concat_unconditional_inputs,
            user_delay_pattern_mask=kwargs.get("user_delay_pattern_mask"),
            assistant_delay_pattern_mask=kwargs.get("assistant_delay_pattern_mask"),
        )

        # The delay pattern masks above had to be built out to the horizon, which is why `max_length` was resolved
        # early. The base loop resolves it again from `max_new_tokens` and reads a `max_length` that is already set
        # as one the caller asked for, so it warns about a clash with itself on every call. Hand it back the
        # unresolved config: it recomputes the same horizon, since by now `input_ids` carries the frame that
        # `concat_unconditional_inputs` prepended.
        generation_config.max_length = None

        # set delay pattern mask for the rest of the generation
        self.generated_user_audio_codes = None
        kwargs["user_delay_pattern_mask"] = (
            user_delay_pattern_mask if user_delay_pattern_mask is not None else kwargs.get("user_delay_pattern_mask")
        )
        kwargs["assistant_delay_pattern_mask"] = (
            assistant_delay_pattern_mask
            if assistant_delay_pattern_mask is not None
            else kwargs.get("assistant_delay_pattern_mask")
        )

        # Undelayed, matching how the loop extends it below.
        self.generated_audio_codes = torch.repeat_interleave(
            undelayed_assistant_audio_codes,
            max(generation_config.num_beams, generation_config.num_return_sequences),
            dim=0,
        )

        # Beam search needs the scores and the beam indices to reorder the audio history afterwards, so both are
        # forced on. They go on the config rather than alongside it: passing generation arguments next to a
        # `generation_config` is deprecated, and doing it here warned on every call.
        # What the caller actually asked for, kept because the checks below distinguish "the caller wanted a dict"
        # from "beam search needed one".
        caller_wants_dict = generation_config.return_dict_in_generate
        return_dict_in_generate = generation_config.num_beams > 1 or caller_wants_dict
        output_scores = generation_config.num_beams > 1 or generation_config.output_scores
        generation_config.return_dict_in_generate = return_dict_in_generate
        generation_config.output_scores = output_scores
        outputs = super().generate(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            generation_config=generation_config,
            kwargs_depth_decoder=kwargs_depth_decoder,
            attention_mask=attention_mask,
            **kwargs,
        )

        if not return_audio_codes:
            if return_dict_in_generate and not caller_wants_dict:
                return outputs.sequences
            return outputs

        # check if outputs is a dict or tokens
        if not return_dict_in_generate:
            output_text_ids = outputs
        else:
            output_text_ids = outputs.sequences

        if generation_config.num_return_sequences > 1:
            assistant_delay_pattern_mask = torch.repeat_interleave(
                assistant_delay_pattern_mask, generation_config.num_return_sequences, dim=0
            )

        if generation_config.num_beams > 1:
            # we need to reorganize self.last_hidden_states and generated audio codes according to the beam_indices

            # Beam indices are of shape `input_length + number_generated_tokens` but actually starts
            # indexing indices at index 0 instead of index `input_length-1`.
            # We thus discard the last `input_length` indices that are never used.
            beam_indices = outputs.beam_indices[:, : -assistant_audio_codes.shape[-1]]

            generated_audio_codes = self.generated_audio_codes[:, :, assistant_audio_codes.shape[-1] :]

            # we've generated audio tokens `number_generated_tokens-1` times, so we use the corresponding beam indices to
            # retrieve the right audio tokens
            expanded_beam_indices = beam_indices[:, :-1].unsqueeze(1).expand(-1, self.num_codebooks, -1)
            generated_audio_codes = torch.gather(generated_audio_codes, dim=0, index=expanded_beam_indices)

            # now, rebuild generated audio codes, this time with the right beam tracking
            assistant_audio_codes = torch.repeat_interleave(
                assistant_audio_codes, generation_config.num_return_sequences, dim=0
            )
            self.generated_audio_codes = torch.cat((assistant_audio_codes, generated_audio_codes), dim=2)

            # use the last beam indice to retrieve the right self.last_hidden_state
            self.last_hidden_state = torch.index_select(self.last_hidden_state, dim=0, index=beam_indices[:, -1])

        # we need to make a last generation with the latest generated tokens
        last_hidden_state = self.last_hidden_state.view(-1, 1, self.last_hidden_state.shape[-1])

        last_generated_audio_codes = self.depth_decoder.generate(
            last_hidden_state=last_hidden_state,
            input_ids=output_text_ids[:, -1:].view(-1, 1),
            **kwargs_depth_decoder,
        )

        # Drop the leading text token, then keep only Moshi's own codebooks.
        last_generated_audio_codes, _ = self._split_predicted_streams(last_generated_audio_codes[:, 1:].unsqueeze(2))

        self.generated_audio_codes = torch.cat([self.generated_audio_codes, last_generated_audio_codes], dim=2)

        # The history is already what the caller wants: the depth decoder emits a whole frame at a time, and the
        # delay is applied only where the codes are fed back in. Applying it here and slicing it off again would
        # not cancel out -- the slice shifts every codebook but the first, so codes that were never delayed come
        # out misaligned by a frame, which is audible as slurred speech and makes a step-by-step caller's output
        # disagree with the same codes decoded in one go.
        output_audio_codes = self.generated_audio_codes

        # Frames the caller did not supply an assistant stream for hold the audio BOS id -- "Moshi has not spoken
        # yet". Mimi has no such code, so returning them would hand the codec an index past its codebooks. They
        # only ever sit at the front, and no real code can collide with the reserved id, so the leading run of
        # them is trimmed.
        is_bos_frame = (output_audio_codes == self.config.audio_vocab_size).all(dim=1).all(dim=0)
        if is_bos_frame.any():
            spoken = (~is_bos_frame).nonzero()
            first_spoken = int(spoken[0]) if spoken.numel() else output_audio_codes.shape[-1]
            output_audio_codes = output_audio_codes[:, :, first_spoken:]

        output_audio_codes = output_audio_codes if return_audio_codes else None
        # Only meaningful when the depth decoder predicted the user stream for frames the caller left open.
        output_user_audio_codes = None
        if return_audio_codes and self.generated_user_audio_codes is not None:
            predicted = self.generated_user_audio_codes[:, :, self._user_supplied_steps :]
            output_user_audio_codes = predicted if predicted.shape[-1] > 0 else None

        if caller_wants_dict:
            return MoshiConditionalGenerationGenerateOutput(
                audio_codes=output_audio_codes, user_audio_codes=output_user_audio_codes, **outputs
            )

        return MoshiConditionalGenerationGenerateOutput(
            sequences=output_text_ids, audio_codes=output_audio_codes, user_audio_codes=output_user_audio_codes
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

            input_ids = model_inputs.pop("input_ids")

            generated_audio_codes = self.depth_decoder.generate(
                last_hidden_state=last_hidden_state,
                input_ids=input_ids.view(-1, 1),
                **kwargs_depth_decoder,
            )

            # the first tokens are text tokens
            generated_audio_codes = generated_audio_codes[:, 1:].unsqueeze(2)

            generated_audio_codes, predicted_user_codes = self._split_predicted_streams(generated_audio_codes)

            # Advance the user stream by one frame. `user_delay_pattern_mask` carries whatever the caller supplied
            # and is `-1` past the end of it, so only the unsupplied frames take the tensor concatenated here --
            # the caller always wins, as upstream's per-slot `provided` guard does. Those frames get the depth
            # decoder's own prediction when it predicts the user stream, and zeros otherwise (in which case
            # `generate` has already refused a stream that falls short of the horizon).
            fill = predicted_user_codes
            if fill is None:
                fill = self.generated_audio_codes.new_zeros(
                    (self.generated_audio_codes.shape[0], self.generated_audio_codes.shape[1], 1)
                )
            user_audio_codes = self.apply_delay_pattern_mask(
                torch.cat([self.generated_audio_codes, fill], dim=2),
                user_delay_pattern_mask,
            )[:, :, -1:]
            if predicted_user_codes is not None:
                self.generated_user_audio_codes = (
                    user_audio_codes
                    if self.generated_user_audio_codes is None
                    else torch.cat([self.generated_user_audio_codes, user_audio_codes], dim=2)
                )
            # Kept undelayed. The delay pattern is a wire format, not the history itself: writing it back into the
            # state would mean a later `build_delay_pattern_mask` -- which shifts whatever it is handed -- shifts
            # the same codes a second time. It is applied where the codes are read instead.
            self.generated_audio_codes = torch.cat([self.generated_audio_codes, generated_audio_codes], dim=2)
            delayed_audio_codes = self.apply_delay_pattern_mask(
                self.generated_audio_codes, assistant_delay_pattern_mask
            )

            inputs_embeds, _, _, _, _, _, _, _ = self._prepare_inputs_embeds_for_generation(
                input_ids,
                assistant_audio_codes=delayed_audio_codes[:, :, -1:],
                user_audio_codes=user_audio_codes,
            )

            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = inputs_embeds

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
        # dirty, but we need to make a last depth_decoder.generate
        self.last_hidden_state = last_hidden_state
        return model_kwargs

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
