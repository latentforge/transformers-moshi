# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Processor class for Moshi."""

import numpy as np
import torch

from ...audio_utils import AudioInput
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring


class MoshiProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            # Moshi is built on Mimi, which operates at 24kHz.
            "sampling_rate": 24000,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


@auto_docstring
class MoshiProcessor(ProcessorMixin):
    valid_processor_kwargs = MoshiProcessorKwargs
    audio_tokenizer_class = "MimiModel"
    default_audio_tokenizer = "kyutai/mimi"

    def __init__(self, feature_extractor=None, tokenizer=None, audio_tokenizer=None, num_codebooks=8):
        r"""
        feature_extractor (`EncodecFeatureExtractor`, *optional*):
            Turns raw audio into the model input the codec expects. It belongs to the codec, so when omitted it is
            fetched from the default codec (`default_audio_tokenizer`), the same as `audio_tokenizer`.
        tokenizer (`PreTrainedTokenizerFast`):
            Moshi's own text tokenizer. Unlike the two codec-derived pieces it is specific to the checkpoint, so
            there is nothing to fall back to -- it is required.
        audio_tokenizer (`MimiModel`, *optional*):
            The Mimi codec. It turns raw audio into the discrete codes Moshi consumes, and turns generated codes
            back into audio. When omitted -- which is what happens loading a checkpoint saved before the codec was
            tracked in the processor config -- the default codec (`default_audio_tokenizer`) is fetched, so
            `MoshiProcessor.from_pretrained(...)` works without hand-assembling the pieces.
        num_codebooks (`int`, *optional*, defaults to 8):
            How many of Mimi's codebooks Moshi uses. Mimi can emit more than Moshi reads, so this is a property of
            the Moshi checkpoint rather than of the codec.
        """
        if tokenizer is None:
            raise ValueError(
                "`tokenizer` is required: it is the model's own text tokenizer, so unlike the codec and its "
                "feature extractor it cannot be filled in from a default."
            )
        # Both codec-derived pieces default to the same codec repo when the caller (or a checkpoint saved without
        # processor metadata, like the original `kmhf/*` conversions) leaves them out. Local imports: they are only
        # reached on this fallback, keeping the module import graph free of a codec dependency otherwise.
        if feature_extractor is None:
            from ..auto import AutoFeatureExtractor

            feature_extractor = AutoFeatureExtractor.from_pretrained(self.default_audio_tokenizer)
        if audio_tokenizer is None:
            from ..mimi import MimiModel

            audio_tokenizer = MimiModel.from_pretrained(self.default_audio_tokenizer)
        self.num_codebooks = num_codebooks
        super().__init__(feature_extractor, tokenizer, audio_tokenizer=audio_tokenizer)

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        audio: AudioInput | None = None,
        assistant_audio: AudioInput | None = None,
        **kwargs: object,
    ) -> BatchFeature:
        r"""
        audio (`AudioInput`, *optional*):
            The user's speech stream. Encoded into `user_audio_codes`.
        assistant_audio (`AudioInput`, *optional*):
            Moshi's own speech stream, used to prompt the model with what it has already said. Encoded into
            `assistant_audio_codes`.

        Moshi runs two audio streams in parallel with the text stream, so it does not take a single `input_values`
        the way a one-stream audio model does. Each stream is passed through the feature extractor and then the
        codec separately, and is named after the speaker it belongs to.
        """
        if text is None and audio is None and assistant_audio is None:
            raise ValueError("You have to specify at least one of `text`, `audio` or `assistant_audio`.")

        merged_kwargs = self._merge_kwargs(
            MoshiProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        data = {}
        if text is not None:
            data.update(self.tokenizer(text, **merged_kwargs["text_kwargs"]))

        for prefix, stream in (("user", audio), ("assistant", assistant_audio)):
            if stream is None:
                continue
            encoded = self.feature_extractor(stream, **merged_kwargs["audio_kwargs"])
            # `padding_mask` is deliberately dropped: the two audio streams are frame-aligned with the text
            # stream, so Moshi has no argument to receive a per-stream audio mask.
            input_values = encoded["input_values"]
            if not isinstance(input_values, torch.Tensor):
                input_values = torch.as_tensor(np.asarray(input_values), dtype=torch.float32)
            input_values = input_values.to(self.audio_tokenizer.device)
            data[f"{prefix}_audio_codes"] = self.audio_tokenizer.encode(
                input_values, num_quantizers=self.num_codebooks
            ).audio_codes

        # Moshi always listens to both streams at once, so a prompt that carries only one of them is incomplete.
        # The missing side was silent, and silence is ordinary audio to Mimi -- it encodes to real codes, not to
        # the reserved "has not spoken yet" id. Filling it here rather than leaving `generate` to stand in that id
        # matters: repeating the reserved id for the length of a prompt is far enough off-distribution to turn the
        # generated text into word salad.
        stream_frames = [codes.shape[-1] for key, codes in data.items() if key.endswith("_audio_codes")]
        if stream_frames:
            batch_size = next(codes.shape[0] for key, codes in data.items() if key.endswith("_audio_codes"))
            for key in ("user_audio_codes", "assistant_audio_codes"):
                if key not in data:
                    data[key] = self.get_silence_audio_codes(max(stream_frames), batch_size=batch_size)

        # Moshi consumes one text token per audio frame, and rejects inputs whose lengths disagree. The original
        # model achieves this by padding the text in between token enunciations, so the text stream is padded out
        # to the number of audio frames here rather than leaving the caller to do it.
        num_frames = max(
            (codes.shape[-1] for key, codes in data.items() if key.endswith("_audio_codes")),
            default=None,
        )
        if num_frames is not None and "input_ids" in data:
            data["input_ids"], data["attention_mask"] = self._pad_text_to_frames(
                data["input_ids"], data.get("attention_mask"), num_frames
            )

        # `_merge_kwargs` distributes `common_kwargs` into each modality, so `return_tensors` is read back from
        # one of them rather than from a `common_kwargs` entry.
        return_tensors = merged_kwargs["text_kwargs"].get("return_tensors")
        return BatchFeature(data=data, tensor_type=return_tensors)

    def _pad_token_id(self) -> int:
        """
        The id used to fill the text stream between enunciations.

        Moshi's published tokenizers carry `<pad>` in their vocabulary but do not declare it as the pad token, so
        `pad_token_id` is `None` there and the vocabulary is consulted as a fallback. A hit is only trusted when it
        is not the unknown-token id, since unknown lookups resolve to that.
        """
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            return pad_id
        pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
        if pad_id is not None and pad_id != self.tokenizer.unk_token_id:
            return pad_id
        raise ValueError(
            "The tokenizer declares no `pad_token_id` and has no `<pad>` in its vocabulary, so the text stream "
            "cannot be padded to match the audio streams."
        )

    def _pad_text_to_frames(self, input_ids, attention_mask, num_frames: int):
        """Right-pad (or truncate) the text stream so it has one token per audio frame."""
        input_ids = torch.as_tensor(input_ids)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None:
            attention_mask = torch.as_tensor(attention_mask)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)

        seq_length = input_ids.shape[-1]
        if seq_length > num_frames:
            raise ValueError(
                f"The text stream is longer than the audio stream ({seq_length} tokens for {num_frames} audio "
                "frames). Moshi needs one text token per audio frame, so shorten the text or lengthen the audio."
            )
        if seq_length < num_frames:
            pad_id = self._pad_token_id()
            padding = input_ids.new_full((input_ids.shape[0], num_frames - seq_length), pad_id)
            input_ids = torch.cat([input_ids, padding], dim=-1)
            if attention_mask is not None:
                # Attended, not masked out. These frames are not absent tokens: Moshi predicts the text stream
                # including its padding, and each frame is locked to an audio frame that is itself attended.
                # Masking them would desynchronise the text stream from the audio it is aligned with.
                attention_mask = torch.cat([attention_mask, torch.ones_like(padding)], dim=-1)

        return input_ids, attention_mask

    def get_silence_audio_codes(self, num_frames: int, batch_size: int = 1) -> "torch.Tensor":
        r"""
        Encode `num_frames` frames of silence into audio codes.

        Moshi is full-duplex: it consumes a user frame at every step, so generating without a live user means
        handing it silence for the whole horizon. Mimi is a streaming codec, so silence does not quantize to a
        single constant -- its convolution state carries across frames and the codes keep changing. Encoding the
        whole span in one go reproduces that, which repeating a single frame would not.

        Args:
            num_frames (`int`):
                How many frames to produce, i.e. how many steps the model will be asked to generate.
            batch_size (`int`, *optional*, defaults to 1):
                How many identical rows to return.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_codebooks, num_frames)`.
        """
        config = self.audio_tokenizer.config
        silence = torch.zeros(
            (1, 1, num_frames * int(config.sampling_rate / config.frame_rate)),
            dtype=self.audio_tokenizer.dtype,
            device=self.audio_tokenizer.device,
        )
        codes = self.audio_tokenizer.encode(silence, num_quantizers=self.num_codebooks).audio_codes
        return codes.expand(batch_size, -1, -1)

    def decode_audio(self, audio_codes: "torch.Tensor", **kwargs) -> "torch.Tensor":
        r"""
        Turn audio codes produced by `MoshiForConditionalGeneration.generate` back into a waveform.

        Args:
            audio_codes (`torch.Tensor` of shape `(batch_size, num_codebooks, sequence_length)`):
                The generated audio codes.

        Returns:
            `torch.Tensor` of shape `(batch_size, 1, audio_sequence_length)`: the decoded waveform.
        """
        return self.audio_tokenizer.decode(audio_codes.to(self.audio_tokenizer.device), **kwargs).audio_values


__all__ = ["MoshiProcessor"]
