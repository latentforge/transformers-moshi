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

    def _embed_audio_codes(self, audio_codes: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Sum the per-codebook audio embeddings.

        `embed_tokens` is a `ModuleList`, so under a device map its entries can land on different devices. Each
        embedding's output is therefore moved to a single device before summing.
        """
        target_device = self.embed_tokens[offset].weight.device
        return sum(
            self.embed_tokens[codebook + offset](audio_codes[:, codebook]).to(target_device)
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
    ):
        user_delay_pattern_mask = None
        assistant_delay_pattern_mask = None

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

        if inputs_embeds is None or apply_delay_pattern_mask:
            if apply_delay_pattern_mask and user_audio_codes is not None:
                user_audio_codes, user_delay_pattern_mask = self.build_delay_pattern_mask(
                    user_audio_codes,
                    bos_token_id=self.config.audio_vocab_size,
                    pad_token_id=self.config.audio_vocab_size,
                    max_length=generation_config.max_length,
                )

            if apply_delay_pattern_mask and assistant_audio_codes is not None:
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

            audio_inputs_embeds = None
            if user_audio_codes is not None and assistant_audio_codes is not None:
                audio_codes = torch.cat([assistant_audio_codes, user_audio_codes], dim=1)
                audio_inputs_embeds = self._embed_audio_codes(audio_codes)
            elif assistant_audio_codes is not None:
                audio_codes = assistant_audio_codes
                audio_inputs_embeds = self._embed_audio_codes(audio_codes)
            elif user_audio_codes is not None:
                audio_codes = user_audio_codes
                audio_inputs_embeds = self._embed_audio_codes(audio_codes, offset=self.num_codebooks)

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
                num_user_frames=1 + (generation_config.max_new_tokens or 0),
            )
        )

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

        # Moshi is full-duplex and consumes a user frame at every step, so the user stream has to reach the end of
        # the horizon. There is no stand-in: a caller with no live user encodes silence for the whole span with
        # `MoshiProcessor.get_silence_audio_codes`.
        if user_audio_codes is not None and user_audio_codes.shape[-1] < generation_config.max_length:
            raise ValueError(
                f"`user_audio_codes` covers {user_audio_codes.shape[-1]} frames but generation runs to "
                f"{generation_config.max_length}. Moshi consumes a user frame at every step, so pass the whole "
                "stream -- `processor.get_silence_audio_codes(num_frames)` produces silence for it."
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
            # `_prepare_attention_mask_for_generation` reads the normalized `_pad_token_tensor` /
            # `_eos_token_tensor`, which `generate` would only set later, so resolve them up front here.
            self._prepare_special_tokens(generation_config, device=input_ids.device)
            attention_mask = self._prepare_attention_mask_for_generation(input_ids, generation_config, kwargs)
        (
            inputs_embeds,
            input_ids,
            user_audio_codes,
            assistant_audio_codes,
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
        )

        # set delay pattern mask for the rest of the generation
        kwargs["user_delay_pattern_mask"] = (
            user_delay_pattern_mask if user_delay_pattern_mask is not None else kwargs.get("user_delay_pattern_mask")
        )
        kwargs["assistant_delay_pattern_mask"] = (
            assistant_delay_pattern_mask
            if assistant_delay_pattern_mask is not None
            else kwargs.get("assistant_delay_pattern_mask")
        )

        self.generated_audio_codes = torch.repeat_interleave(
            assistant_audio_codes, max(generation_config.num_beams, generation_config.num_return_sequences), dim=0
        )

        return_dict_in_generate = generation_config.num_beams > 1 or generation_config.return_dict_in_generate
        output_scores = generation_config.num_beams > 1 or generation_config.output_scores
        outputs = super().generate(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            generation_config=generation_config,
            kwargs_depth_decoder=kwargs_depth_decoder,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
            attention_mask=attention_mask,
            **kwargs,
        )

        if not return_audio_codes:
            if return_dict_in_generate and not generation_config.return_dict_in_generate:
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

        # Drop the leading text token, then keep only Moshi's own codebooks (see the same slice in the loop above).
        last_generated_audio_codes = last_generated_audio_codes[:, 1:].unsqueeze(2)[:, : self.num_codebooks]

        self.generated_audio_codes = torch.cat([self.generated_audio_codes, last_generated_audio_codes], dim=2)

        # apply the pattern mask to the final audio ids
        output_audio_codes = self.apply_delay_pattern_mask(self.generated_audio_codes, assistant_delay_pattern_mask)

        # Revert the delay pattern. Codebook 0 is unshifted and the rest are shifted right by one, so undoing it is
        # a slice; the leading frame holds the BOS row and is dropped with it. Filtering on the pad/bos id instead
        # would only work when the codes span exactly `max_length`, which is not the case when `generate` is driven
        # one step at a time.
        output_audio_codes = torch.cat([output_audio_codes[:, :1, 1:-1], output_audio_codes[:, 1:, 2:]], dim=1)

        output_audio_codes = output_audio_codes if return_audio_codes else None

        if generation_config.return_dict_in_generate:
            return MoshiConditionalGenerationGenerateOutput(audio_codes=output_audio_codes, **outputs)

        return MoshiConditionalGenerationGenerateOutput(sequences=output_text_ids, audio_codes=output_audio_codes)

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

            # A depth decoder trained on both streams predicts `dep_q = 2 * num_codebooks` codebooks, Moshi's own
            # first and the user's after (`forward` feeds them in that order). Only Moshi's half is fed back as the
            # generated audio; the user stream is supplied by the caller.
            generated_audio_codes = generated_audio_codes[:, : self.num_codebooks]

            # Advance the user stream by one frame. `user_delay_pattern_mask` holds the stream the caller passed in
            # and it reaches the end of the horizon, so the placeholder concatenated here is always overwritten by
            # `apply_delay_pattern_mask`; it only exists to give the mask a tensor of the right length.
            placeholder = self.generated_audio_codes.new_zeros(
                (self.generated_audio_codes.shape[0], self.generated_audio_codes.shape[1], 1)
            )
            user_audio_codes = self.apply_delay_pattern_mask(
                torch.cat([self.generated_audio_codes, placeholder], dim=2),
                user_delay_pattern_mask,
            )[:, :, -1:]
            self.generated_audio_codes = self.apply_delay_pattern_mask(
                torch.cat([self.generated_audio_codes, generated_audio_codes], dim=2), assistant_delay_pattern_mask
            )

            inputs_embeds, _, _, _, _, _, _ = self._prepare_inputs_embeds_for_generation(
                input_ids,
                assistant_audio_codes=self.generated_audio_codes[:, :, -1:],
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
    # Copied from transformers.models.musicgen.modeling_musicgen.MusicgenForCausalLM.apply_delay_pattern_mask
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

        # if one or two of the three required inputs have been passed, throws an error
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
