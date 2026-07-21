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
"""Weight transforms from the original Kyutai layout to the `transformers` one.

Kept apart from `convert_moshi_to_hf.py` so that reading a checkpoint in the original layout does not drag in the
conversion script's command-line dependencies: this module imports nothing beyond `torch` and the loading
primitives. `MoshiForConditionalGeneration` and `PersonaPlexForConditionalGeneration` are both released that way.
"""

from typing import Any

import torch

from ...core_model_loading import (
    ConversionOps,
    MergeModulelist,
    WeightConverter,
    WeightRenaming,
)


def _single_tensor(input_dict: dict[str, Any]) -> torch.Tensor:
    """Grab the only tensor collected by a one-to-one/one-to-many conversion."""
    tensors = next(iter(input_dict.values()))
    return tensors[0] if isinstance(tensors, list) else tensors


class SqueezeGain(ConversionOps):
    """
    Moshi stores the `RMSNorm` gains of the original implementation as `alpha` buffers of shape `(1, 1, hidden_size)`,
    while the HF `nn.Parameter` is 1D.
    """

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, Any], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        return {target_patterns[0]: _single_tensor(input_dict).squeeze()}


class SplitFusedQkv(ConversionOps):
    """
    Split the original fused `self_attn.in_proj_weight` into the three `q/k/v` projections.

    The two decoders need different treatment, and they are told apart by the (already renamed) target key:

    - main decoder: a plain `(3 * hidden_size, hidden_size)` matrix, split along dim 0. `q` and `k` additionally
      need the sliced-rotary permutation, because the original implementation applies RoPE on interleaved pairs
      while `transformers` applies it on split halves. `v` is *not* permuted, as RoPE never touches it.
    - depth decoder: the projection is per-codebook, so the matrix is first viewed as
      `(num_codebooks, 3 * inner, hidden)` and split along dim 1. RoPE is not used there, hence no permutation.
    """

    @staticmethod
    def _permute_for_sliced_rope(weight: torch.Tensor, num_heads: int, dim1: int, dim2: int) -> torch.Tensor:
        return weight.view(num_heads, dim1 // num_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, Any],
        source_patterns: list[str],
        target_patterns: list[str],
        config,
        full_layer_name: str,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        mixed_qkv = _single_tensor(input_dict)

        if full_layer_name.startswith("depth_decoder."):
            # The depth decoder predicts its own number of codebooks (`dep_q`), which is not necessarily the
            # parent's per-stream count.
            mixed_qkv = mixed_qkv.view(config.depth_decoder_config.num_codebooks, -1, mixed_qkv.shape[-1])
            qkv_dim = mixed_qkv.size(1) // 3
            query, key, value = (
                mixed_qkv[:, :qkv_dim],
                mixed_qkv[:, qkv_dim : qkv_dim * 2],
                mixed_qkv[:, qkv_dim * 2 :],
            )
        else:
            qkv_dim = mixed_qkv.size(0) // 3
            query, key, value = mixed_qkv[:qkv_dim], mixed_qkv[qkv_dim : qkv_dim * 2], mixed_qkv[qkv_dim * 2 :]
            num_heads = int(config.hidden_size // config.head_dim)
            key_value_head_dim = config.num_key_value_heads * config.head_dim
            query = self._permute_for_sliced_rope(query, num_heads, config.hidden_size, config.hidden_size)
            key = self._permute_for_sliced_rope(
                key, config.num_key_value_heads, key_value_head_dim, config.hidden_size
            )

        query_target, key_target, value_target = target_patterns
        return {
            query_target: query.contiguous(),
            key_target: key.contiguous(),
            value_target: value.contiguous(),
        }


class SplitPerCodebook(ConversionOps):
    """
    Reshape a depth-decoder projection stored as a single `(num_codebooks * inner, hidden)` matrix into the
    per-codebook `(num_codebooks, inner, hidden)` parameter used by `transformers`.
    """

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, Any],
        source_patterns: list[str],
        target_patterns: list[str],
        config,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        return {target_patterns[0]: tensor.view(config.depth_decoder_config.num_codebooks, -1, tensor.shape[-1])}


# Pure renamings, applied (in order) to every checkpoint key before any `WeightConverter` runs. They are anchored
# with `^` wherever the original name is a top-level module, which is what keeps `depformer_text_emb` from being
# swallowed by the `depformer_emb`/`text_emb` rules — the old substring chain could not express that and needed a
# fixup pass afterwards.
WEIGHT_RENAMINGS = [
    # Top-level modules of the main decoder.
    WeightRenaming(source_patterns=r"^out_norm\.", target_patterns="model.norm."),
    WeightRenaming(source_patterns=r"^text_emb\.", target_patterns="model.embed_tokens."),
    WeightRenaming(source_patterns=r"^text_linear\.", target_patterns="lm_head."),
    WeightRenaming(source_patterns=r"^emb\.", target_patterns="embed_tokens."),
    WeightRenaming(source_patterns=r"^transformer\.", target_patterns="model."),
    # Top-level modules of the depth decoder. `MoshiDepthDecoderForCausalLM` is a `MoshiDepthDecoderModel` under
    # `model` plus `lm_heads`, so everything but the heads lands under `depth_decoder.model.` directly.
    WeightRenaming(source_patterns=r"^depformer_text_emb\.", target_patterns="depth_decoder.model.text_embed_tokens."),
    WeightRenaming(source_patterns=r"^depformer_emb\.", target_patterns="depth_decoder.model.embed_tokens."),
    WeightRenaming(source_patterns=r"^depformer\.", target_patterns="depth_decoder.model."),
    # Layer internals, shared by both decoders.
    WeightRenaming(source_patterns=r"\.gating\.linear_in\.", target_patterns=".mlp.fc1."),
    WeightRenaming(source_patterns=r"\.gating\.linear_out\.", target_patterns=".mlp.fc2."),
    WeightRenaming(source_patterns=r"\.self_attn\.out_proj\.", target_patterns=".self_attn.o_proj.linear."),
    WeightRenaming(source_patterns=r"\.norm1\.", target_patterns=".input_layernorm."),
    WeightRenaming(source_patterns=r"\.norm2\.", target_patterns=".post_attention_layernorm."),
    WeightRenaming(source_patterns=r"\.layer_scale_1\.", target_patterns=".self_attn_layer_scale."),
    WeightRenaming(source_patterns=r"\.layer_scale_2\.", target_patterns=".mlp_layer_scale."),
]

# Actual weight surgery. These run after every renaming above, so their patterns are expressed in the renamed
# namespace, except for the two original module lists (`depformer_in`, `linears`) that no renaming touches.
WEIGHT_CONVERTERS = [
    # `alpha` gains are stored with two leading singleton dims.
    WeightConverter(source_patterns="alpha", target_patterns="weight", operations=[SqueezeGain()]),
    # Fused qkv, for both decoders (`SplitFusedQkv` branches on the target key).
    WeightConverter(
        source_patterns=r"self_attn\.in_proj_weight",
        target_patterns=[
            "self_attn.q_proj.linear.weight",
            "self_attn.k_proj.linear.weight",
            "self_attn.v_proj.linear.weight",
        ],
        operations=[SplitFusedQkv()],
    ),
    # The depth decoder's output projection is per-codebook; the main decoder's is a plain matrix, hence the scoping.
    WeightConverter(
        source_patterns=r"self_attn\.o_proj\.linear\.weight",
        target_patterns="self_attn.o_proj.linear.weight",
        operations=[SplitPerCodebook()],
    ),
    # The depth decoder's gating MLP is one module per codebook in the original checkpoint; `transformers` stacks
    # them into a single 3D parameter. The main decoder has a single gating module, renamed above.
    WeightConverter(
        source_patterns="gating.*.linear_in.weight",
        target_patterns="mlp.fc1.weight",
        operations=[MergeModulelist(dim=0)],
    ),
    WeightConverter(
        source_patterns="gating.*.linear_out.weight",
        target_patterns="mlp.fc2.weight",
        operations=[MergeModulelist(dim=0)],
    ),
    # Same story for the per-codebook input projections and language-modelling heads of the depth decoder.
    WeightConverter(
        source_patterns="depformer_in.*.weight",
        target_patterns="depth_decoder.model.input_projections.weight",
        operations=[MergeModulelist(dim=0)],
    ),
    WeightConverter(
        source_patterns="linears.*.weight",
        target_patterns="depth_decoder.lm_heads.weight",
        operations=[MergeModulelist(dim=0)],
    ),
]

# `self_attn.o_proj.linear.weight` exists in both decoders but only needs reshaping in the depth one; restrict that
# converter to keys under `depth_decoder.` so the main decoder keeps the plain renaming path.
WEIGHT_CONVERTERS[2].scope_prefix = "depth_decoder"
WEIGHT_CONVERTERS[2].base_model_prefix = ""
