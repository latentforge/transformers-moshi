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
"""PyTorch Moshi model."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict
from torch.nn import CrossEntropyLoss

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, ModelOutput
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..mistral.modeling_mistral import MistralModel
from .generation_moshi import MoshiGenerationMixin


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="kmhf/hf-moshiko")
@strict
class MoshiDepthConfig(PreTrainedConfig):
    r"""
    input_size (`int`, *optional*, defaults to 4096):
        Dimensionality of the input hidden states. Used to connect the main decoder to the depth decoder.
    audio_vocab_size (`int`, *optional*, defaults to 2048):
        Vocabulary size of the audio part of model. Defines the number of different tokens that can be
        represented by the `audio_codes` passed when calling the Moshi models.
    ffn_dim (`int`, *optional*, defaults to 5632):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the depth decoder block. Must be even.

    Example:

    ```python
    >>> from transformers import (
    ...     MoshiDepthConfig,
    ...     MoshiDepthDecoderModel,
    ... )

    >>> configuration = MoshiDepthConfig()

    >>> # Initializing a MoshiDepthDecoderModel (with random weights) from the kmhf/hf-moshiko style configuration
    >>> model = MoshiDepthDecoderModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "moshi_depth"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 32000
    hidden_size: int = 1024
    input_size: int = 4096
    num_hidden_layers: int = 6
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    audio_vocab_size: int = 2048
    max_position_embeddings: int = 17
    hidden_act: str = "silu"
    head_dim: int | None = None
    initializer_range: float = 0.02
    use_cache: bool = True
    attention_dropout: float | int = 0.0
    ffn_dim: int = 5632
    rms_norm_eps: float = 1e-8
    num_codebooks: int = 16
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = (
            self.num_key_value_heads if self.num_key_value_heads is not None else self.num_attention_heads
        )
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        # The depth decoder runs along the codebook axis: position 0 holds the text token and positions 1..N hold
        # the codebooks it predicts, so the number of positions follows from `num_codebooks`.
        derived_positions = self.num_codebooks + 1
        if self.max_position_embeddings != derived_positions:
            logger.warning(
                f"`max_position_embeddings` is derived from `num_codebooks` for the depth decoder: overriding "
                f"{self.max_position_embeddings} with {derived_positions} (= `num_codebooks` + 1 for the text token)."
            )
            self.max_position_embeddings = derived_positions
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.ffn_dim % 2 == 1:
            raise ValueError(f"`ffn_dim={self.ffn_dim}` must be even.")


@auto_docstring(checkpoint="kmhf/hf-moshiko")
@strict
class MoshiConfig(PreTrainedConfig):
    r"""
    audio_vocab_size (`int`, *optional*):
        Vocabulary size of the audio part of model. Defines the number of different tokens that can be
        represented by the `audio_codes` passed when calling the Moshi models.
    ffn_dim (`int`, *optional*, defaults to 22528):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the main decoder block. Must be even.
    audio_encoder_config (`PreTrainedConfig | dict`, *optional*):
        Configuration for the audio encoder.
    depth_decoder_config (`PreTrainedConfig | dict`, *optional*):
        Configuration for the depth decoder.

    Example:

    ```python
    >>> from transformers import (
    ...     MoshiConfig,
    ...     MoshiForConditionalGeneration,
    ... )

    >>> configuration = MoshiConfig()

    >>> # Initializing a MoshiForConditionalGeneration (with random weights) from the kmhf/hf-moshiko style configuration
    >>> model = MoshiForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("kmhf/hf-moshiko")

    >>> # loading model and config from pretrained folder
    >>> moshi_config = MoshiConfig.from_pretrained("kmhf/hf-moshiko")
    >>> model = MoshiForConditionalGeneration.from_pretrained("kmhf/hf-moshiko", config=moshi_config)
    ```"""

    model_type = "moshi"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"audio_encoder_config": AutoConfig, "depth_decoder_config": MoshiDepthConfig}
    # The projections are wrapped in `MoshiLinear`, hence the extra `.linear`.
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj.linear": "colwise",
        "layers.*.self_attn.k_proj.linear": "colwise",
        "layers.*.self_attn.v_proj.linear": "colwise",
        "layers.*.self_attn.o_proj.linear": "rowwise",
        # `fc1` is a fused `[gate; up]` projection that `MoshiGatingMLP` splits with a `view`, so its output
        # has to be gathered before the split (a plain `colwise` shard would hand whole gate/up halves to
        # different ranks). `fc2` then takes a replicated input. Same pairing as Phi3's `gate_up_proj`.
        "layers.*.mlp.fc1": "colwise_gather_output",
        "layers.*.mlp.fc2": "rowwise_split_input",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    audio_vocab_size: int | None = None
    max_position_embeddings: int = 3000
    rope_parameters: RopeParameters | dict | None = None
    hidden_act: str = "silu"
    head_dim: int | None = None
    initializer_range: float = 0.02
    use_cache: bool = True
    sliding_window: int | None = 3000
    attention_dropout: float | int = 0.0
    ffn_dim: int = 22528
    rms_norm_eps: float = 1e-8
    num_codebooks: int = 8
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    audio_encoder_config: dict | PreTrainedConfig | None = None
    depth_decoder_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = (
            self.num_key_value_heads if self.num_key_value_heads is not None else self.num_attention_heads
        )
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads

        if isinstance(self.audio_encoder_config, dict):
            audio_encoder_model_type = self.audio_encoder_config.pop("model_type", "mimi")
            self.audio_encoder_config = AutoConfig.for_model(audio_encoder_model_type, **self.audio_encoder_config)
        elif self.audio_encoder_config is None:
            self.audio_encoder_config = AutoConfig.for_model("mimi")

        self.audio_vocab_size = (
            self.audio_encoder_config.codebook_size if self.audio_vocab_size is None else self.audio_vocab_size
        )

        # The depth decoder consumes the main decoder's hidden states and codebooks, so these four values must
        # mirror the parent. `None` is treated as an empty dict so the defaults get synced too, instead of
        # silently keeping `MoshiDepthConfig`'s own (e.g. `input_size=4096`) and failing at runtime.
        if self.depth_decoder_config is None:
            self.depth_decoder_config = {}
        if isinstance(self.depth_decoder_config, dict):
            # These three name the same quantity on both configs, so they are mirrored from the parent. A
            # conflicting value used to be overwritten silently, which hid genuine misconfiguration.
            mirrored = {
                "audio_vocab_size": self.audio_vocab_size,
                "input_size": self.hidden_size,
                "vocab_size": self.vocab_size,
            }
            for key, parent_value in mirrored.items():
                given = self.depth_decoder_config.get(key)
                if given is not None and given != parent_value:
                    parent_name = "hidden_size" if key == "input_size" else key
                    logger.warning(
                        f"`depth_decoder_config['{key}']={given}` conflicts with `{parent_name}={parent_value}` and "
                        f"is overridden with {parent_value}. The depth decoder consumes the main decoder's hidden "
                        "states and codebooks, so the two must agree."
                    )
            self.depth_decoder_config.update(mirrored)
            # `num_codebooks` is deliberately not mirrored: it does not mean the same thing on both configs. On the
            # parent it is the number of codebooks *per audio stream*; on the depth decoder it is how many
            # codebooks are *predicted* (`dep_q` upstream, versus `n_q = 2 * num_codebooks` in total). They happen
            # to coincide in the released checkpoints, which predict only Moshi's own stream because the user-side
            # heads were dropped, but a model trained to predict both streams has twice as many. So the parent's
            # value is only a default here.
            self.depth_decoder_config.setdefault("num_codebooks", 2 * self.num_codebooks)
            # Resolved through `sub_configs` rather than named directly, so a subclass that swaps in its own depth
            # config class gets it built here instead of silently ending up with Moshi's.
            self.depth_decoder_config = self.sub_configs["depth_decoder_config"](**self.depth_decoder_config)

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.ffn_dim % 2 == 1:
            raise ValueError(f"`ffn_dim={self.ffn_dim}` must be even.")

        if self.num_codebooks > self.audio_encoder_config.num_codebooks:
            raise ValueError(
                f"`num_codebooks={self.num_codebooks}` is greater than the maximum number of codebooks that the audio encoder can deal with ({self.audio_encoder_config.num_codebooks}). Please lower it."
            )

    @property
    def sampling_rate(self):
        return self.audio_encoder_config.sampling_rate

    @classmethod
    def from_audio_encoder_config(
        cls,
        audio_encoder_config: PreTrainedConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`MoshiConfig`] (or a derived class) from an audio encoder configuration.

        Returns:
            [`MoshiConfig`]: An instance of a configuration object
        """

        return cls(
            audio_encoder_config=audio_encoder_config.to_dict(),
            **kwargs,
        )


@auto_docstring(
    custom_intro="""
    `MoshiForConditionalGeneration` outputs.
    """
)
@dataclass
class MoshiConditionalGenerationOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `text_labels` is provided):
        Text language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the text language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    depth_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `audio_labels` is provided):
        Audio language modeling loss (for next-token prediction).
    audio_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the audio language modeling heads.
    depth_past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Past key-values of the depth decoder.
    depth_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Hidden states of the depth decoder
    depth_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        Depth decoder's Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
        heads.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    depth_loss: torch.FloatTensor | None = None
    audio_logits: torch.FloatTensor | None = None
    depth_past_key_values: Cache | None = None
    depth_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    depth_attentions: tuple[torch.FloatTensor, ...] | None = None


class MoshiRMSNorm(LlamaRMSNorm):
    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        # Moshi scales by the weight in float32 (before casting back), unlike Llama which casts first.
        return (self.weight.float() * x).to(input_dtype)


class MoshiFlexibleLinear(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super().__init__()
        # Stack the weights for N layers into a single tensor (num_layers, output_size, input_size)
        self.weight = nn.Parameter(torch.randn(num_layers, output_size, input_size))

    def forward(self, x, layer_idx=None):
        """
        `MoshiFlexibleLinear` creates one linear layer per codebook. There's multiple ways to use it.
        In the default case, `sequence_length=num_layers`, so each element of the sequence will be matmul to the weights corresponding to its index on the sequence.

        For more advanced cases, one can specify which codebook's layer(s) to use with `layer_idx`.
        If `layer_idx` indicates a single integer, all of the element of the sequence will be matmul to this single codebook's layer.
        But if `layer_idx` is a tensor of shape `(seq_length,)`, it will matmul each i-th element of the input sequence to the corresponding layer `weight[i]`.


        Args:
            x (`torch.FloatTensor): input to the layer of shape `(batch, num_layers, embed_dim)` or of shape `(batch, seq_length, embed_dim)`
            layer_idx (`torch.Tensor`, *optional*):
                Can be used to specify which codebook's layers(s) to use.
                If it's a tensor of shape `(seq_length,)`, will matmul each element of the sequence to the corresponding weights.
                But if `layer_idx` is a tensor of shape `(seq_length,)`, it will matmul each i-th element of the input sequence to the corresponding layer `weight[i]`.
        """

        # Use torch.gather to select the corresponding weights for each sample
        # (codebooks, output_size, hidden_size)
        selected_weights = torch.index_select(self.weight, 0, layer_idx) if layer_idx is not None else self.weight

        # (1, codebooks, hidden_size, output_size)
        selected_weights = selected_weights.transpose(1, 2)[None, :, :, :]

        # (batch_size, codebooks, 1, hidden_size) x (1, codebooks, hidden_size, output_size)
        # -> (batch_size, codebooks, 1, output_size)
        x = torch.matmul(x[:, :, None, :], selected_weights)

        # (batch_size, codebooks, output_size)
        return x.squeeze(2)


class MoshiLinear(nn.Module):
    def __init__(self, input_dim, output_dim, num_codebooks, use_flexible_linear=False):
        super().__init__()

        self.use_flexible_linear = use_flexible_linear

        if not use_flexible_linear:
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.linear = MoshiFlexibleLinear(input_dim, output_dim, num_layers=num_codebooks)

    def forward(self, x, layer_idx=None):
        if self.use_flexible_linear:
            return self.linear(x, layer_idx)
        else:
            return self.linear(x)


class MoshiRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class MoshiGatingMLP(nn.Module):
    def __init__(self, config, use_flexible_linear=False):
        super().__init__()

        self.activation_fn = ACT2FN[config.hidden_act]
        ffn_dim = config.ffn_dim
        hidden_size = config.hidden_size
        num_layers = config.num_codebooks if use_flexible_linear else 1
        if num_layers == 1:
            self.fc1 = nn.Linear(hidden_size, ffn_dim, bias=False)
            self.fc2 = nn.Linear(ffn_dim // 2, hidden_size, bias=False)
        else:
            self.fc1 = MoshiFlexibleLinear(hidden_size, ffn_dim, num_layers)
            self.fc2 = MoshiFlexibleLinear(ffn_dim // 2, hidden_size, num_layers)

    def forward(self, hidden_states: torch.Tensor, layer_idx: int | None = None) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states) if layer_idx is None else self.fc1(hidden_states, layer_idx)

        batch_size, sequence_length, _ = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, sequence_length, 2, -1)
        hidden_states = self.activation_fn(hidden_states[..., 0, :]) * hidden_states[..., 1, :]
        hidden_states = self.fc2(hidden_states) if layer_idx is None else self.fc2(hidden_states, layer_idx)
        return hidden_states


class MoshiAttention(LlamaAttention):
    def __init__(self, config: MoshiConfig, layer_idx: int | None = None, use_flexible_linear=False):
        super().__init__(config, layer_idx)
        # Moshi keeps the explicit `1 / sqrt(head_dim)` scaling to stay bit-exact with the reference impl.
        self.scaling = 1 / math.sqrt(self.head_dim)
        # `MoshiLinear` wraps either a plain `nn.Linear` (main decoder) or a per-codebook
        # `MoshiFlexibleLinear` (depth decoder), selected by `use_flexible_linear`.
        self.q_proj = MoshiLinear(
            config.hidden_size, config.num_attention_heads * self.head_dim, config.num_codebooks, use_flexible_linear
        )
        self.k_proj = MoshiLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, config.num_codebooks, use_flexible_linear
        )
        self.v_proj = MoshiLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, config.num_codebooks, use_flexible_linear
        )
        self.o_proj = MoshiLinear(
            config.num_attention_heads * self.head_dim, config.hidden_size, config.num_codebooks, use_flexible_linear
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        codebook_idx: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states, codebook_idx).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states, codebook_idx).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states, codebook_idx).view(hidden_shape).transpose(1, 2)

        # rotary embeddings are not used in the depth decoder, where `position_embeddings` is None
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output, codebook_idx)
        return attn_output, attn_weights


class MoshiDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: MoshiConfig, layer_idx: int, use_flexible_linear: bool):
        super().__init__(config, layer_idx)
        self.use_flexible_linear = use_flexible_linear
        self.self_attn = MoshiAttention(config=config, layer_idx=layer_idx, use_flexible_linear=use_flexible_linear)
        self.mlp = MoshiGatingMLP(config, use_flexible_linear)
        self._attn_implementation = config._attn_implementation

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        codebook_idx: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            codebook_idx=codebook_idx,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = (
            self.mlp(hidden_states) if not self.use_flexible_linear else self.mlp(hidden_states, codebook_idx)
        )
        hidden_states = residual + hidden_states

        return hidden_states


@auto_docstring
class MoshiPreTrainedModel(PreTrainedModel):
    config: MoshiConfig
    base_model_prefix = "model"
    input_modalities = ("audio", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["MoshiDecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": MoshiDecoderLayer,
        "attentions": MoshiAttention,
    }

    main_input_name = "input_ids"

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, MoshiFlexibleLinear):
            init.normal_(module.weight)


def get_past_seen_tokens(past_key_values: Cache | None) -> int:
    """
    How many codebooks the depth decoder has already cached, as a Python `int`.

    A static cache -- which is what the depth decoder generates with -- keeps its length as a 0-d tensor, so that
    the count stays on device for cudagraphs. Here it is only ever used as an offset: to pick an embedding out of a
    `ModuleList` and to index `input_ids`. Left as a tensor it makes those indices tensors too, and `torch.compile`
    cannot trace an index it has to read the value of. Converting once per forward costs a single sync, against one
    per position had the loop read it back with `.item()`.
    """
    if past_key_values is None:
        return 0
    past_seen_tokens = past_key_values.get_seq_length()
    return int(past_seen_tokens) if torch.is_tensor(past_seen_tokens) else past_seen_tokens


def get_codebook_idx(
    input_ids: torch.LongTensor | None,
    inputs_embeds: torch.FloatTensor | None,
    past_seen_tokens: int,
) -> torch.Tensor:
    """Position `i` of the depth decoder's sequence is codebook `i` of a single main-decoder timestep."""
    sequence = inputs_embeds if inputs_embeds is not None else input_ids
    return torch.arange(sequence.shape[1], device=sequence.device) + past_seen_tokens


@auto_docstring(
    custom_intro="""
    Transformer depth decoder consisting of *config.num_hidden_layers* layers, each one a [`MoshiDecoderLayer`].

    It runs along the codebook axis rather than the time axis: position `i` of its sequence is codebook `i` of a
    single timestep of the main decoder. It therefore uses neither rotary embeddings nor a final norm, and it
    embeds its inputs from several sources (a text embedding for the first position, one audio embedding per
    codebook for the rest, plus a per-codebook projection of the main decoder's hidden state).
    """
)
class MoshiDepthDecoderModel(MoshiPreTrainedModel):
    config: MoshiDepthConfig

    def __init__(self, config: MoshiDepthConfig):
        super().__init__(config)

        self.text_embed_tokens = nn.Embedding(config.vocab_size + 1, config.hidden_size)

        # the last codebook is never used as input
        self.embed_tokens = nn.ModuleList(
            [nn.Embedding(config.audio_vocab_size + 1, config.hidden_size) for _ in range(config.num_codebooks - 1)]
        )

        self.input_projections = MoshiFlexibleLinear(config.input_size, config.hidden_size, config.num_codebooks)

        # the depth decoder does not use rotary embeddings, so no `position_embeddings` are passed to the layers
        self.layers = nn.ModuleList(
            [
                MoshiDecoderLayer(config, layer_idx, use_flexible_linear=True)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self._attn_implementation = config._attn_implementation
        self.gradient_checkpointing = False
        self.config = config

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        last_hidden_state: torch.FloatTensor | None = None,
        attention_mask: torch.BoolTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens. The first element of the sequence must be the text token associated to
            the audio codebooks. The rest of the elements must be flattened audio codebooks.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the main decoder. Used to contextualize
            `input_ids`.
        """
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        past_seen_tokens = get_past_seen_tokens(past_key_values)
        codebook_idx = get_codebook_idx(input_ids, inputs_embeds, past_seen_tokens)

        if position_ids is None:
            position_ids = codebook_idx.unsqueeze(0)

        # If inputs_embeds is provided, it has the priority over input_ids, which won't be used
        if inputs_embeds is None:
            inputs_embeds = []
            # Which embedding a position uses is a Python-level choice -- `embed_tokens` is a `ModuleList` with one
            # entry per codebook -- so the loop runs over plain ints. Reading them back off `codebook_idx` with
            # `.item()` would force a device sync per position and break the compiled graph, for indices that are
            # already known here: it is built as `arange(seq_len) + past_seen_tokens`.
            for position_idx in range(past_seen_tokens, past_seen_tokens + input_ids.shape[1]):
                if position_idx == 0:
                    inputs_embeds.append(self.text_embed_tokens(input_ids[:, [position_idx]]))
                else:
                    inputs_embeds.append(
                        self.embed_tokens[(position_idx - 1)](input_ids[:, [position_idx - past_seen_tokens]])
                    )

            inputs_embeds = torch.cat(inputs_embeds, dim=1)

        inputs_embeds += self.input_projections(last_hidden_state, codebook_idx)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # decoder layers
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                codebook_idx=codebook_idx,
            )

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


@auto_docstring(
    custom_intro="""
    The Moshi depth decoder with a per-codebook audio language modelling head on top.
    """
)
class MoshiDepthDecoderForCausalLM(MoshiPreTrainedModel, GenerationMixin):
    config: MoshiDepthConfig
    # `lm_heads` emits audio logits, so it is unrelated to the text embeddings of the backbone.
    _tied_weights_keys = None
    _tp_plan = None
    _pp_plan = None

    def __init__(self, config: MoshiDepthConfig):
        super().__init__(config)
        self.model = MoshiDepthDecoderModel(config)
        self.lm_heads = MoshiFlexibleLinear(config.hidden_size, config.audio_vocab_size, config.num_codebooks)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        last_hidden_state: torch.FloatTensor | None = None,
        attention_mask: torch.BoolTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens. The first element of the sequence must be the text token associated to
            the audio codebooks. The rest of the elements must be flattened audio codebooks.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the main decoder. Used to contextualize
            `input_ids`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the audio language modeling loss. Indices should either be in
            `[0, ..., config.audio_vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to
            `-100` are ignored (masked).
        """
        # `lm_heads` is indexed per codebook, so recompute the same indices the backbone uses. This must happen
        # before the backbone call, which advances `past_key_values`.
        past_seen_tokens = get_past_seen_tokens(past_key_values)
        codebook_idx = get_codebook_idx(input_ids, inputs_embeds, past_seen_tokens)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            last_hidden_state=last_hidden_state,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            position_ids=position_ids,
            **kwargs,
        )

        logits = self.lm_heads(outputs.last_hidden_state, codebook_idx)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            loss_fct = CrossEntropyLoss()

            labels = labels.masked_fill(labels == self.config.audio_vocab_size, -100).reshape(-1)
            labels = labels.to(logits.device)
            loss = loss_fct(logits.reshape(-1, self.config.audio_vocab_size), labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class MoshiModel(MistralModel, MoshiPreTrainedModel):
    # Inherits from `MistralModel` rather than `LlamaModel` for its mask selection: Moshi's temporal transformer
    # limits attention to `sliding_window` (3000 frames, i.e. 240s at 12.5Hz) exactly like the original
    # `StreamingMultiheadAttention`, which masks with `delta < context` and backs it with a ring KV cache. Only
    # this transformer is windowed -- the depth decoder is full-causal upstream too, since `lm.py` overwrites
    # `depformer_context` with `None`.
    def __init__(self, config: MoshiConfig):
        super().__init__(config)
        # Moshi reserves one extra id on top of the text vocabulary (used as the audio-only/BOS position)
        self.embed_tokens = nn.Embedding(config.vocab_size + 1, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                MoshiDecoderLayer(config, layer_idx, use_flexible_linear=False)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )


@auto_docstring(
    custom_intro="""
    The Moshi decoder model with a text language modelling head on top. Only usable for text.
    """
)
class MoshiForCausalLM(LlamaForCausalLM, MoshiPreTrainedModel):
    input_modalities = ("text",)
    # Moshi's text embedding has one extra id (`vocab_size + 1`) compared to `lm_head`, so the two can never be tied.
    _tied_weights_keys = None

    def forward(self, **super_kwargs) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MoshiForCausalLM

        >>> model = MoshiForCausalLM.from_pretrained("kmhf/hf-moshiko")
        >>> tokenizer = AutoTokenizer.from_pretrained("kmhf/hf-moshiko")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        ```"""
        return super().forward(**super_kwargs)


@auto_docstring(
    custom_intro="""
    The original Moshi model with an audio encoder, a Moshi depth decoder and a Moshi decoder, for speech-to-speech.
    """
)
class MoshiForConditionalGeneration(MoshiPreTrainedModel, MoshiGenerationMixin):
    config: MoshiConfig
    # Checkpoints published before the codec moved to `MoshiProcessor` carry Mimi's weights under
    # `audio_encoder.`. The model no longer holds the codec, so those keys are expected to go unused.
    _keys_to_ignore_on_load_unexpected = [r"^audio_encoder\."]
    output_modalities = ("audio", "text")
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    # Only the text (main) decoder is tensor/pipeline parallel; the depth decoder's per-codebook
    # `MoshiFlexibleLinear` weights are 3D and do not map onto the standard colwise/rowwise strategies.
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: MoshiConfig):
        super().__init__(config)
        # We have 2 * num_codebooks audio embedding layers because we have the user input channel and the model output channel.
        self.embed_tokens = nn.ModuleList(
            [nn.Embedding(config.audio_vocab_size + 1, config.hidden_size) for _ in range(2 * config.num_codebooks)]
        )
        self.model = MoshiModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.depth_decoder = MoshiDepthDecoderForCausalLM._from_config(config.depth_decoder_config)

        # `num_codebooks` is the width of a single audio stream. The depth decoder predicts either just Moshi's own
        # stream, or Moshi's followed by the user's -- the released checkpoints do the former because the user-side
        # heads were dropped, a model trained on both streams does the latter. Anything else cannot be split into
        # streams, so reject it here rather than silently keeping the leading half.
        self.num_codebooks = config.num_codebooks
        predicted_codebooks = config.depth_decoder_config.num_codebooks
        if predicted_codebooks not in (self.num_codebooks, 2 * self.num_codebooks):
            raise ValueError(
                f"The depth decoder predicts {predicted_codebooks} codebooks, which is neither one stream "
                f"({self.num_codebooks}) nor both ({2 * self.num_codebooks})."
            )
        self.predicts_user_stream = predicted_codebooks == 2 * self.num_codebooks
        self.post_init()

    def get_depth_decoder(self):
        return self.depth_decoder

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.BoolTensor | None = None,
        user_audio_codes: torch.Tensor | None = None,
        assistant_audio_codes: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        text_labels: torch.LongTensor | None = None,
        audio_labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> MoshiConditionalGenerationOutputWithPast:
        r"""
        user_audio_codes (`torch.Tensor `of shape `(batch_size, num_codebooks, sequence_length), *optional*):
            The audio codes used as audio user prompt for the generation, as produced by [`MoshiProcessor`].
        assistant_audio_codes (`torch.Tensor `of shape `(batch_size, num_codebooks, sequence_length), *optional*):
            The audio codes used as audio Moshi prompt for the generation, as produced by [`MoshiProcessor`].
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
        >>> from transformers import MoshiForConditionalGeneration

        >>> model = MoshiForConditionalGeneration.from_pretrained("kmhf/hf-moshiko")
        >>> inputs = model.get_unconditional_inputs()

        >>> logits = model(**inputs).logits
        >>> logits.shape  # (batch_size, sequence_length, text_vocab_size)
        torch.Size([1, 1, 32000])
        ```"""
        # Resolve the output flags here so the decoder's output-capturing sees a concrete boolean instead of
        # `None`, which would otherwise disable capture even when the config enables it.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        kwargs_depth_decoder = {
            argument[len("depth_decoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("depth_decoder_")
        }

        # If inputs_embeds is provided, it has the priority over input_ids and audio_codes, which won't be used
        if inputs_embeds is None:
            audio_codes = torch.cat([assistant_audio_codes, user_audio_codes], dim=1)

            if input_ids is None and audio_codes is None:
                raise ValueError("You must provide at least one of `input_ids`, `inputs_embeds` and the audio codes.")

            if input_ids is not None:
                inputs_embeds = self.model.embed_tokens(input_ids)

            if audio_codes is not None:
                audio_inputs_embeds = self._embed_audio_codes(audio_codes)
                inputs_embeds = (
                    audio_inputs_embeds
                    if inputs_embeds is None
                    else audio_inputs_embeds + inputs_embeds.to(audio_inputs_embeds.device)
                )

        # Decode
        decoder_outputs: BaseModelOutputWithPast = self.model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **kwargs_decoder,
        )

        decoder_last_hidden_state = decoder_outputs.last_hidden_state
        logits = self.lm_head(decoder_last_hidden_state)

        loss = None
        if text_labels is not None:
            loss = self.loss_function(logits=logits, labels=text_labels, vocab_size=self.config.vocab_size, **kwargs)

        depth_decoder_outputs = None
        final_loss = loss
        if text_labels is not None and audio_labels is not None:
            # To use depth decoder forward here, we actually need oracle input ids since we're supposed to pass the true input ids

            audio_labels = self.build_delay_pattern_mask(
                audio_labels,
                bos_token_id=self.config.audio_vocab_size,
                pad_token_id=self.config.audio_vocab_size,
                max_length=audio_labels.shape[-1] + 1,
            )[0]

            # (batch_size, sequence_length) -> (batch_size * sequence_length, 1)
            text_labels = text_labels.view(-1, 1)

            # (batch_size, num_codebooks, sequence_length) -> (batch_size * sequence_length, num_codebooks)
            audio_labels = audio_labels.transpose(1, 2).reshape(-1, audio_labels.shape[1])

            depth_input_ids = torch.cat([text_labels, audio_labels], dim=1)
            # keep the last codebook out of input_ids
            depth_input_ids = depth_input_ids[:, :-1]

            # (batch_size, sequence_length, dim) -> (batch_size * sequence_length, 1, dim)
            decoder_last_hidden_state = decoder_last_hidden_state.view(-1, 1, decoder_last_hidden_state.shape[-1])

            # No `attention_mask` here: the depth decoder runs on a flattened `(batch * sequence_length)`
            # batch whose sequence axis is the codebook axis, which is never padded, so the main decoder's
            # `(batch, sequence_length)` mask neither applies nor has a compatible batch size.
            depth_decoder_outputs = self.depth_decoder(
                last_hidden_state=decoder_last_hidden_state,
                input_ids=depth_input_ids,
                labels=audio_labels,
                **kwargs_depth_decoder,
            )

            final_loss += depth_decoder_outputs.loss

        return MoshiConditionalGenerationOutputWithPast(
            # `final_loss` is the text loss plus, when `audio_labels` are given, the depth decoder loss
            loss=final_loss,
            logits=logits,
            last_hidden_state=decoder_last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            depth_loss=None if depth_decoder_outputs is None else depth_decoder_outputs.loss,
            audio_logits=None if depth_decoder_outputs is None else depth_decoder_outputs.logits,
            depth_past_key_values=None if depth_decoder_outputs is None else depth_decoder_outputs.past_key_values,
            depth_hidden_states=None if depth_decoder_outputs is None else depth_decoder_outputs.hidden_states,
            depth_attentions=None if depth_decoder_outputs is None else depth_decoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def freeze_depth_decoder(self):
        """
        Freeze the depth encoder weights.
        """
        for param in self.depth_decoder.parameters():
            param.requires_grad = False
        self.depth_decoder._requires_grad = False


__all__ = [
    "MoshiConfig",
    "MoshiDepthConfig",
    "MoshiDepthDecoderForCausalLM",
    "MoshiDepthDecoderModel",
    "MoshiForCausalLM",
    "MoshiForConditionalGeneration",
    "MoshiModel",
    "MoshiPreTrainedModel",
]
