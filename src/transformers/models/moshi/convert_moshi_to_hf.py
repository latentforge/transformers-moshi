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
from transformers.core_model_loading import convert_and_load_state_dict_in_model
from transformers.modeling_utils import LoadStateDictConfig
from transformers.models.moshi.conversion_moshi import WEIGHT_CONVERTERS, WEIGHT_RENAMINGS


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
    )

    generation_config = GenerationConfig(
        do_sample=True,
        temp=0.7,
        top_k=25,
        # No `cache_implementation`: it is deprecated for anything but the static caches, and which layers are
        # sliding is inferred from the config now, so naming it here only earns a warning on every generate.
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
