# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Convert Moshi checkpoints."""

import argparse
import re
from typing import Any

import safetensors
import sentencepiece
import torch

from transformers import (
    AutoFeatureExtractor,
    GenerationConfig,
    MimiModel,  # initial audio encoder
    MoshiConfig,
    MoshiForConditionalGeneration,
    MoshiProcessor,
    PreTrainedTokenizerFast,
    logging,
)
from transformers.convert_slow_tokenizer import MoshiConverter
from transformers.core_model_loading import (
    ConversionOps,
    MergeModulelist,
    WeightConverter,
    WeightRenaming,
    convert_and_load_state_dict_in_model,
)
from transformers.modeling_utils import LoadStateDictConfig


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.mimi")


def assert_param_count(model_1, model_2):
    count_1 = sum(p[1].numel() for p in model_1.named_parameters() if "final_proj" not in p[0])
    count_2 = sum(p[1].numel() for p in model_2.named_parameters() if "final_proj" not in p[0])
    assert count_1 == count_2, f"{model_1.__class__}: {count_1} != {model_2.__class__}: {count_2}"


def param_count(model):
    return sum(p[1].numel() for p in model.named_parameters() if "final_proj" not in p[0])


def _grab_best_device(use_gpu=True):
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)


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


def _convert_model(state_dict, hf_model, device):
    load_config = LoadStateDictConfig(
        device_map={"": "cpu"},
        dtype=torch.bfloat16,
        weight_mapping=[*WEIGHT_RENAMINGS, *WEIGHT_CONVERTERS],
    )
    loading_info, _ = convert_and_load_state_dict_in_model(hf_model, state_dict, load_config, tp_plan=None)

    if loading_info.conversion_errors:
        raise ValueError(f"conversion errors: {loading_info.conversion_errors}")
    if loading_info.unexpected_keys:
        raise ValueError(f"extra keys found: {loading_info.unexpected_keys}")
    if loading_info.missing_keys:
        raise ValueError(f"missing keys: {loading_info.missing_keys}")
    if loading_info.mismatched_keys:
        raise ValueError(f"mismatched keys: {loading_info.mismatched_keys}")

    n_params = param_count(hf_model)
    logger.info(f"model loaded: {round(n_params / 1e6, 1)}M params")

    hf_model.eval()
    hf_model.to(device)
    del state_dict

    return hf_model


@torch.no_grad()
def convert_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    mimi_repo_id,
    config_path=None,
    repo_id=None,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    device = _grab_best_device()

    # Kept in float32: the codec's weights are no longer merged into the Moshi checkpoint, so this instance only
    # supplies the config and the silence codes below. Quantizing silence in bfloat16 on CPU picks different
    # entries for the deeper residual codebooks than the float32 codec `MoshiProcessor` runs at inference time,
    # which would bake codes into the config that the processor never reproduces.
    mimi_model = MimiModel.from_pretrained(mimi_repo_id)

    original_checkpoint = safetensors.torch.load_file(checkpoint_path)
    if "best_state" in original_checkpoint:
        # we might have a training state saved, in which case discard the yaml results and just retain the weights
        original_checkpoint = original_checkpoint["best_state"]

    if config_path is not None:
        config = MoshiConfig.from_pretrained(config_path)
    else:
        audio_encoder_config = mimi_model.config
        config = MoshiConfig.from_audio_encoder_config(audio_encoder_config)

    # How many codebooks the depth decoder predicts (`dep_q` upstream) is a property of the checkpoint, not of the
    # parent: the released weights drop the user-side heads, whereas a model trained on both streams keeps them.
    # Count the heads rather than assuming, so either kind converts.
    num_depth_heads = sum(1 for key in original_checkpoint if re.fullmatch(r"linears\.\d+\.weight", key))
    if num_depth_heads == 0:
        raise ValueError("Found no `linears.{i}.weight` in the checkpoint, so the depth decoder cannot be sized.")
    config.depth_decoder_config.num_codebooks = num_depth_heads
    config.depth_decoder_config.max_position_embeddings = num_depth_heads + 1

    model = MoshiForConditionalGeneration(config).to(torch.bfloat16)

    depth_decoder_generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.8,
        top_k=250,
        # The depth decoder emits one token per codebook *it predicts* (`dep_q`) plus the leading text token, so
        # this follows the depth config rather than the parent's per-stream count.
        min_length=config.depth_decoder_config.num_codebooks + 1,
        max_length=config.depth_decoder_config.num_codebooks + 1,
        cache_implementation="sliding_window",
    )

    generation_config = GenerationConfig(
        do_sample=True,
        temp=0.7,
        top_k=25,
        cache_implementation="sliding_window",
        pad_token_id=config.vocab_size,
        bos_token_id=config.vocab_size,
    )
    generation_config.depth_decoder_config = depth_decoder_generation_config.to_diff_dict()

    model.generation_config = generation_config

    # The codec is no longer part of the model: it lives in `MoshiProcessor` as the `audio_tokenizer`, so its
    # weights are not merged into the checkpoint here.
    model = _convert_model(original_checkpoint, model, device)

    # `save_original_format=True` (the default) would write the depth decoder back in the pre-split flat layout
    # via the reverse conversion mapping. Newly converted checkpoints should use the current layout.
    model.save_pretrained(pytorch_dump_folder_path, save_original_format=False)

    if repo_id:
        print("Pushing to the hub...")
        model.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument(
        "--tokenizer_vocab_path", required=False, default=None, type=str, help="Path to original tokenizer vocab file"
    )
    parser.add_argument("--mimi_repo_id", required=True, default=None, type=str, help="Repository id to HF Mimi.")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the Hugging Face hub."
    )

    args = parser.parse_args()

    convert_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.mimi_repo_id,
        args.config_path,
        args.push_to_hub,
    )

    # Assemble the processor. It owns the Mimi codec as its `audio_tokenizer`, so the codec is referenced by repo
    # id rather than copied into the checkpoint.
    if args.tokenizer_vocab_path is None:
        raise ValueError("`--tokenizer_vocab_path` is required to build the `MoshiProcessor`.")

    original_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer_vocab_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=MoshiConverter(args.tokenizer_vocab_path).converted(),
        chat_template=None,
        model_input_names=["input_ids", "attention_mask"],
        clean_up_tokenization_spaces=False,
        # Declared as tokens, not ids: passing `*_token_id` leaves `pad_token_id` and friends unset on the saved
        # tokenizer, which is why the published checkpoints report `pad_token_id is None` even though `<pad>` is
        # in their vocabulary.
        unk_token=original_tokenizer.id_to_piece(original_tokenizer.unk_id()),
        bos_token=original_tokenizer.id_to_piece(original_tokenizer.bos_id()),
        eos_token=original_tokenizer.id_to_piece(original_tokenizer.eos_id()),
        pad_token=original_tokenizer.id_to_piece(original_tokenizer.pad_id()),
    )

    processor = MoshiProcessor(
        feature_extractor=AutoFeatureExtractor.from_pretrained(args.mimi_repo_id),
        tokenizer=tokenizer,
        audio_tokenizer=MimiModel.from_pretrained(args.mimi_repo_id),
        num_codebooks=MoshiConfig.from_pretrained(args.pytorch_dump_folder_path).num_codebooks,
    )
    processor.save_pretrained(args.pytorch_dump_folder_path)

    if args.push_to_hub:
        print("Pushing the processor to the hub...")
        processor.push_to_hub(args.push_to_hub)
