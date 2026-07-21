# Copyright 2026 NVIDIA CORPORATION & AFFILIATES, Kyutai and The HuggingFace Inc. team. All rights reserved.
#
# PersonaPlex is NVIDIA's fine-tune of Kyutai's Moshi; the reference implementation it is ported from is MIT
# licensed and derives from Moshi's own release.
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
"""PyTorch PersonaPlex model."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ..auto import AutoConfig
from ..moshi.configuration_moshi import MoshiConfig, MoshiDepthConfig
from ..moshi.generation_moshi import MoshiGenerationMixin
from ..moshi.modeling_moshi import (
    MoshiDepthDecoderForCausalLM,
    MoshiDepthDecoderModel,
    MoshiForCausalLM,
    MoshiForConditionalGeneration,
    MoshiModel,
    MoshiPreTrainedModel,
)


@auto_docstring(checkpoint="nvidia/personaplex-7b-v1")
@strict
class PersonaPlexDepthConfig(MoshiDepthConfig):
    r"""
    input_size (`int`, *optional*, defaults to 4096):
        Dimensionality of the input hidden states. Used to connect the main decoder to the depth decoder.
    audio_vocab_size (`int`, *optional*, defaults to 2048):
        Vocabulary size of the audio part of model. Defines the number of different tokens that can be
        represented by the `audio_codes` passed when calling the PersonaPlex models.
    """

    model_type = "personaplex_depth"


@auto_docstring(checkpoint="nvidia/personaplex-7b-v1")
@strict
class PersonaPlexConfig(MoshiConfig):
    r"""
    audio_vocab_size (`int`, *optional*):
        Vocabulary size of the audio part of model. Defines the number of different tokens that can be
        represented by the `audio_codes` passed when calling the PersonaPlex models.
    audio_encoder_config (`PreTrainedConfig | dict`, *optional*):
        Configuration for the audio encoder.
    depth_decoder_config (`PreTrainedConfig | dict`, *optional*):
        Configuration for the depth decoder.
    """

    model_type = "personaplex"
    sub_configs = {"audio_encoder_config": AutoConfig, "depth_decoder_config": PersonaPlexDepthConfig}

    depth_decoder_config: dict | PreTrainedConfig | None = None


@auto_docstring
class PersonaPlexPreTrainedModel(MoshiPreTrainedModel):
    config: PersonaPlexConfig
    _no_split_modules = ["PersonaPlexDecoderLayer"]


class PersonaPlexDepthDecoderModel(MoshiDepthDecoderModel):
    config: PersonaPlexDepthConfig


class PersonaPlexDepthDecoderForCausalLM(MoshiDepthDecoderForCausalLM):
    config: PersonaPlexDepthConfig


class PersonaPlexModel(MoshiModel):
    pass


class PersonaPlexForCausalLM(MoshiForCausalLM):
    def forward(self, **super_kwargs) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, PersonaPlexForCausalLM

        >>> model = PersonaPlexForCausalLM.from_pretrained("nvidia/personaplex-7b-v1")
        >>> tokenizer = AutoTokenizer.from_pretrained("nvidia/personaplex-7b-v1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        ```"""
        return super().forward(**super_kwargs)


@auto_docstring(
    custom_intro="""
    The PersonaPlex model with a depth decoder and a main decoder, for speech-to-speech.

    Use [`PersonaPlexProcessor`] to build the persona prefix the checkpoint expects; see its docstring for what the
    prefix is made of.
    """
)
class PersonaPlexForConditionalGeneration(MoshiForConditionalGeneration, MoshiGenerationMixin):
    config: PersonaPlexConfig

    def forward(self, **super_kwargs) -> PersonaPlexConditionalGenerationOutputWithPast:  # noqa: F821
        r"""
        user_audio_codes (`torch.Tensor `of shape `(batch_size, num_codebooks, sequence_length), *optional*):
            The audio codes used as audio user prompt for the generation, as produced by [`PersonaPlexProcessor`].
        assistant_audio_codes (`torch.Tensor `of shape `(batch_size, num_codebooks, sequence_length), *optional*):
            The audio codes used as audio PersonaPlex prompt for the generation, as produced by [`PersonaPlexProcessor`].
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If `input_ids` and `inputs_embeds` are both unset, `inputs_embeds` takes the value
            of `inputs_embeds`.
        text_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for text language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        audio_labels (`torch.LongTensor` of shape `(batch_size, num_codebooks, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.audio_vocab_size]`

        Example:

        ```python
        >>> from transformers import AutoProcessor, PersonaPlexForConditionalGeneration

        >>> processor = AutoProcessor.from_pretrained("nvidia/personaplex-7b-v1")
        >>> model = PersonaPlexForConditionalGeneration.from_pretrained("nvidia/personaplex-7b-v1")

        >>> inputs = processor(text_prompt="You are Jane, a patient teacher.", return_tensors="pt")

        >>> logits = model(**inputs).logits
        >>> logits.shape  # (batch_size, sequence_length, text_vocab_size)
        ```"""
        return super().forward(**super_kwargs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        r"""
        PersonaPlex is released in the original Kyutai weight layout (`transformer.*`, `depformer.*`, fused
        `in_proj_weight`, one module per codebook) rather than a `transformers` one, so the checkpoint is converted
        as it is read. The transforms are Moshi's own; they are registered here rather than in `conversion_mapping`'s
        table because that table holds the deltas between successive `transformers` layouts, whereas this is a whole
        foreign format, and registering it on the way in keeps its import off every other model.
        """
        from ...conversion_mapping import register_checkpoint_conversion_mapping
        from ..moshi.conversion_moshi import WEIGHT_CONVERTERS, WEIGHT_RENAMINGS

        register_checkpoint_conversion_mapping(cls.__name__, [*WEIGHT_RENAMINGS, *WEIGHT_CONVERTERS], overwrite=True)
        # Reach the base implementation directly: `super().from_pretrained` cannot be used here (the modular
        # converter would try to inline a parent method that only exists on the library base class).
        return PreTrainedModel.from_pretrained.__func__(cls, *args, **kwargs)


__all__ = [
    "PersonaPlexConfig",
    "PersonaPlexDepthConfig",
    "PersonaPlexDepthDecoderForCausalLM",
    "PersonaPlexDepthDecoderModel",
    "PersonaPlexForCausalLM",
    "PersonaPlexForConditionalGeneration",
    "PersonaPlexModel",
    "PersonaPlexPreTrainedModel",
]
