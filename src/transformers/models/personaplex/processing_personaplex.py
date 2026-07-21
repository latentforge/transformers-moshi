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
"""Processor class for PersonaPlex."""

import numpy as np
import torch

from ...audio_utils import AudioInput
from ...feature_extraction_utils import BatchFeature
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring
from ..moshi.processing_moshi import MoshiProcessor, MoshiProcessorKwargs


# The persona prefix the checkpoint was trained on. Every stage is one frame per column, and the three streams stay
# frame-aligned throughout:
#
#     stage          | text            | assistant stream    | user stream
#     ---------------|-----------------|---------------------|-------------
#     voice prompt   | padding         | the voice sample    | sine
#     silence        | padding         | silence             | sine
#     text prompt    | one token/frame | silence             | sine
#     silence        | padding         | silence             | sine
#
# The user stream is a 440 Hz tone rather than silence for the whole prefix, which is what the model was shown
# during training to mark "this is the prompt, not a turn from the user".
#
# Both fillers are single frames repeated for as long as the stage lasts, and they are the reference
# implementation's own constants rather than something re-encoded here. Encoding the waveforms instead does not
# reproduce them: Mimi is a streaming codec, so its output drifts frame to frame, and no frame of a freshly encoded
# 440 Hz tone matches `SINE_TOKENS` (5 of 8 codebooks at best) or of silence matches `SILENCE_TOKENS` (6 of 8).
# Whatever conditions produced them are not recoverable from the released code, and since the checkpoint was
# validated against inference that feeds exactly these values, they are used verbatim.
SILENCE_TOKENS = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]
SINE_TOKENS = [430, 1268, 381, 1611, 1095, 1495, 56, 472]
SILENCE_DURATION = 0.5  # seconds of silence on each side of the text prompt
VOICE_PROMPT_LUFS = -24.0  # the loudness the voice sample is normalized to

# The released PersonaPlex repository carries its tokenizer as a raw SentencePiece model rather than as a
# `transformers` one, and no processor config at all, so `from_pretrained` builds the processor from those assets.
PERSONAPLEX_SPM_NAME = "tokenizer_spm_32k_3.model"


@auto_docstring
class PersonaPlexProcessor(MoshiProcessor):
    r"""
    PersonaPlex is Moshi fine-tuned to be steered by a persona: a voice sample that fixes *how* it speaks and a
    `<system>`-tagged instruction that fixes *who* it is. Both are supplied as a prefix rather than through extra
    model inputs, so this processor only differs from [`MoshiProcessor`] in that it can build that prefix.
    """

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        audio_tokenizer=None,
        num_codebooks=8,
        silence_codes=None,
        sine_codes=None,
    ):
        r"""
        feature_extractor (`EncodecFeatureExtractor`, *optional*):
            The codec's feature extractor. Filled in from the default codec when omitted, like [`MoshiProcessor`].
        audio_tokenizer (`MimiModel`, *optional*):
            The Mimi codec. It turns raw audio into the discrete codes PersonaPlex consumes, and turns generated
            codes back into audio. Defaults to the base codec (`default_audio_tokenizer`) when omitted, like
            [`MoshiProcessor`].
        num_codebooks (`int`, *optional*, defaults to 8):
            How many of Mimi's codebooks each audio stream uses. Note this is per stream: the depth decoder of a
            checkpoint that predicts both streams emits twice as many.
        silence_codes (`list[int]`, *optional*):
            The codes standing for one frame of silence on the assistant stream during the persona prefix.
        sine_codes (`list[int]`, *optional*):
            The codes standing for one frame of the tone the user stream carries during the persona prefix.

        Both default to the reference implementation's constants. They are properties of the checkpoint's prompt
        format rather than of the codec, which is why they live here and are saved with the processor: a
        checkpoint trained with different fillers only has to ship different values.
        """
        super().__init__(feature_extractor, tokenizer, audio_tokenizer, num_codebooks=num_codebooks)
        self.silence_codes = list(SILENCE_TOKENS if silence_codes is None else silence_codes)
        self.sine_codes = list(SINE_TOKENS if sine_codes is None else sine_codes)
        for name, codes in (("silence_codes", self.silence_codes), ("sine_codes", self.sine_codes)):
            if len(codes) != self.num_codebooks:
                raise ValueError(
                    f"`{name}` has {len(codes)} entries but the processor reads {self.num_codebooks} codebooks. "
                    "They stand for one frame of audio, so there has to be exactly one code per codebook."
                )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Load a processor, assembling one from the original assets when the repository has no `transformers` one.

        The released PersonaPlex repository ships a raw SentencePiece model and a codec checkpoint in the original
        Kyutai layout, but no processor config, so the ordinary path finds nothing to load. Rather than making
        callers wire the three components together by hand, the pieces are built here from what the repository
        actually contains. A repository that does carry a processor config takes the ordinary path untouched.
        """
        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except (OSError, ValueError):
            pass

        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError

        from ...convert_slow_tokenizer import MoshiConverter
        from ...tokenization_utils_tokenizers import PreTrainedTokenizerFast

        try:
            spm_path = hf_hub_download(str(pretrained_model_name_or_path), PERSONAPLEX_SPM_NAME)
        except (HfHubHTTPError, OSError) as error:
            raise OSError(
                f"{pretrained_model_name_or_path} has neither a processor config nor a `{PERSONAPLEX_SPM_NAME}` to "
                "build one from."
            ) from error

        import sentencepiece

        original = sentencepiece.SentencePieceProcessor(spm_path)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=MoshiConverter(spm_path).converted(),
            chat_template=None,
            model_input_names=["input_ids", "attention_mask"],
            clean_up_tokenization_spaces=False,
            unk_token=original.id_to_piece(original.unk_id()),
            bos_token=original.id_to_piece(original.bos_id()),
            eos_token=original.id_to_piece(original.eos_id()),
            pad_token=original.id_to_piece(original.pad_id()),
        )

        # Only the tokenizer is built here -- it is the one piece specific to the checkpoint. The feature
        # extractor and the codec both come from the default codec repo, which `MoshiProcessor.__init__` fills in
        # when they are left out.
        return cls(tokenizer=tokenizer, **kwargs)

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        audio: AudioInput | None = None,
        assistant_audio: AudioInput | None = None,
        voice_prompt: AudioInput | None = None,
        text_prompt: str | None = None,
        **kwargs: object,
    ) -> BatchFeature:
        r"""
        voice_prompt (`AudioInput`, *optional*):
            A sample of the voice the model should speak in. Loudness-normalized and fed on the assistant stream.
        text_prompt (`str`, *optional*):
            The persona instruction, e.g. `"You are Jane, a patient teacher."`. Wrapped in `<system>` tags if it is
            not already.

        Returns the same keys as [`MoshiProcessor`]. When either prompt is given, the persona prefix is built and
        the ordinary inputs are appended to it, so the result can be passed straight to `generate`.
        """
        if voice_prompt is None and text_prompt is None:
            return super().__call__(text=text, audio=audio, assistant_audio=assistant_audio, **kwargs)

        merged_kwargs = self._merge_kwargs(
            MoshiProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        return_tensors = merged_kwargs["text_kwargs"].get("return_tensors")

        prefix = self._build_persona_prefix(voice_prompt, text_prompt, merged_kwargs)

        if text is not None or audio is not None or assistant_audio is not None:
            rest = super().__call__(text=text, audio=audio, assistant_audio=assistant_audio, **kwargs)
            prefix = {
                key: torch.cat([value, torch.as_tensor(rest[key]).to(value.device)], dim=-1)
                for key, value in prefix.items()
                if key in rest
            } | {key: value for key, value in prefix.items() if key not in rest}

            # The appended call fills in whichever audio stream the caller left out, but it only produces text when
            # the caller passed some. Without this the text channel would stop at the end of the prefix while the
            # audio ran on, and the model rejects streams of unequal length.
            num_frames = max(codes.shape[-1] for key, codes in prefix.items() if key.endswith("_audio_codes"))
            prefix["input_ids"], prefix["attention_mask"] = self._pad_text_to_frames(
                prefix["input_ids"], prefix.get("attention_mask"), num_frames
            )

        return BatchFeature(data=prefix, tensor_type=return_tensors)

    def _build_persona_prefix(self, voice_prompt, text_prompt, merged_kwargs) -> dict:
        """Lay the four stages out as one frame-aligned block of `input_ids` and per-stream audio codes."""
        codec_config = self.audio_tokenizer.config
        sampling_rate = codec_config.sampling_rate
        frame_size = int(sampling_rate / codec_config.frame_rate)
        silence_frames = int(SILENCE_DURATION * codec_config.frame_rate)

        # The assistant stream is the voice sample followed by silence; the number of frames it occupies sets how
        # long the first stage is.
        if voice_prompt is not None:
            voice_waveform = self._prepare_voice_prompt(voice_prompt, sampling_rate, merged_kwargs)
            voice_frames = int(np.ceil(voice_waveform.shape[-1] / frame_size))
            # Pad the tail so the sample lands on a frame boundary.
            voice_waveform = np.pad(voice_waveform, (0, voice_frames * frame_size - voice_waveform.shape[-1]))
        else:
            voice_waveform = np.zeros(0, dtype=np.float32)
            voice_frames = 0

        prompt_ids = self._encode_text_prompt(text_prompt)
        prompt_frames = prompt_ids.shape[-1]

        total_frames = voice_frames + silence_frames + prompt_frames + silence_frames

        # The two fillers are the stored constants repeated per frame, which is what the reference implementation
        # feeds. Only the voice sample is actually encoded.
        assistant_codes = self._repeat_frame(self.silence_codes, total_frames)
        if voice_frames:
            assistant_codes[:, :, :voice_frames] = self._encode_waveform(voice_waveform)

        data = {
            "assistant_audio_codes": assistant_codes,
            "user_audio_codes": self._repeat_frame(self.sine_codes, total_frames),
        }

        # The text stream is padding everywhere except the prompt, which gets one token per frame.
        input_ids = prompt_ids.new_full((1, total_frames), self._text_padding_id())
        input_ids[:, voice_frames + silence_frames : voice_frames + silence_frames + prompt_frames] = prompt_ids
        data["input_ids"] = input_ids
        data["attention_mask"] = torch.ones_like(input_ids)

        return data

    def _text_padding_id(self) -> int:
        """
        The id the text stream carries on frames that enunciate nothing.

        The released tokenizer has `<pad>` in its vocabulary but does not declare it as `pad_token`, so
        `pad_token_id` is `None` and the token has to be looked up by name as a fallback.
        """
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
        if pad_id is None or pad_id == self.tokenizer.unk_token_id:
            raise ValueError(
                "The tokenizer declares no `pad_token_id` and has no `<pad>` token, so the text stream of the "
                "persona prefix cannot be padded to the audio frames."
            )
        return pad_id

    def _encode_text_prompt(self, text_prompt: str | None) -> "torch.Tensor":
        """Tokenize the persona instruction, adding the `<system>` tags the checkpoint was trained with."""
        if text_prompt is None:
            return torch.empty((1, 0), dtype=torch.long)
        cleaned = text_prompt.strip()
        if not (cleaned.startswith("<system>") and cleaned.endswith("<system>")):
            cleaned = f"<system> {cleaned} <system>"
        # `<system>` is not a special token: the checkpoint reuses Moshi's tokenizer unchanged, so the tags go
        # through the ordinary vocabulary and take up several frames, exactly as in training.
        return self.tokenizer(cleaned, return_tensors="pt", add_special_tokens=False)["input_ids"]

    def _prepare_voice_prompt(self, voice_prompt, sampling_rate: int, merged_kwargs) -> "np.ndarray":
        """Run the voice sample through the feature extractor and normalize its loudness."""
        encoded = self.feature_extractor(voice_prompt, **merged_kwargs["audio_kwargs"])
        waveform = np.asarray(encoded["input_values"], dtype=np.float32).reshape(-1)

        try:
            import pyloudnorm
        except ImportError as error:
            raise ImportError(
                "The voice prompt is loudness-normalized before being encoded, which requires `pyloudnorm`. "
                "Install it with `pip install pyloudnorm`."
            ) from error

        meter = pyloudnorm.Meter(sampling_rate)
        loudness = meter.integrated_loudness(waveform)
        return np.asarray(pyloudnorm.normalize.loudness(waveform, loudness, VOICE_PROMPT_LUFS), dtype=np.float32)

    def _repeat_frame(self, codes: list[int], num_frames: int) -> "torch.Tensor":
        """Hold one frame of codes for `num_frames`, as `(1, num_codebooks, num_frames)`."""
        frame = torch.tensor(codes, dtype=torch.long, device=self.audio_tokenizer.device)
        return frame.view(1, -1, 1).expand(1, -1, num_frames).contiguous()

    def _encode_waveform(self, waveform: "np.ndarray") -> "torch.Tensor":
        tensor = torch.as_tensor(waveform, dtype=torch.float32, device=self.audio_tokenizer.device)
        tensor = tensor.reshape(1, 1, -1).to(self.audio_tokenizer.dtype)
        return self.audio_tokenizer.encode(tensor, num_quantizers=self.num_codebooks).audio_codes


__all__ = ["PersonaPlexProcessor"]
