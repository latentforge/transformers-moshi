# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import tempfile
import unittest

import numpy as np

from transformers import AutoProcessor, EncodecFeatureExtractor, MimiModel, MoshiProcessor, PreTrainedTokenizerFast
from transformers.testing_utils import require_tokenizers, require_torch


SAMPLING_RATE = 24000
NUM_CODEBOOKS = 4


@require_torch
@require_tokenizers
class MoshiProcessorTest(unittest.TestCase):
    def get_processor(self):
        from tokenizers import Tokenizer, models, pre_tokenizers

        # A tiny stand-in for Moshi's tokenizer: these tests cover the processor, not the tokenizer itself.
        backend = Tokenizer(models.WordLevel({"<unk>": 0, "hello": 1, "world": 2}, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            pad_token="<unk>",
            model_input_names=["input_ids", "attention_mask"],
        )
        feature_extractor = EncodecFeatureExtractor(sampling_rate=SAMPLING_RATE)
        from ..mimi.test_modeling_mimi import MimiModelTester

        # Reuse Mimi's own tiny test config: it is known-valid, and these tests cover the processor, not the codec.
        codec = MimiModel(MimiModelTester(self).get_config()).eval()
        return MoshiProcessor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            audio_tokenizer=codec,
            num_codebooks=NUM_CODEBOOKS,
        )

    def get_audio(self, seconds=1):
        return np.random.randn(SAMPLING_RATE * seconds).astype(np.float32)

    def test_both_audio_streams_are_named_after_their_speaker(self):
        processor = self.get_processor()
        inputs = processor(text="hello", audio=self.get_audio(), assistant_audio=self.get_audio())

        # Moshi runs two audio streams in parallel, so neither is named `input_values`.
        self.assertEqual(
            sorted(inputs.keys()), ["assistant_audio_codes", "attention_mask", "input_ids", "user_audio_codes"]
        )
        self.assertEqual(inputs["user_audio_codes"].shape[:2], (1, NUM_CODEBOOKS))
        self.assertEqual(inputs["assistant_audio_codes"].shape[:2], (1, NUM_CODEBOOKS))

    def test_each_modality_is_optional(self):
        processor = self.get_processor()

        self.assertEqual(sorted(processor(text="hello").keys()), ["attention_mask", "input_ids"])
        # Either audio argument on its own still yields both streams: Moshi listens to itself and to the user at
        # every frame, so the side the caller left out is filled with encoded silence rather than left absent.
        both = ["assistant_audio_codes", "user_audio_codes"]
        self.assertEqual(sorted(processor(audio=self.get_audio()).keys()), both)
        self.assertEqual(sorted(processor(assistant_audio=self.get_audio()).keys()), both)

    def test_missing_stream_is_encoded_silence_not_the_reserved_id(self):
        processor = self.get_processor()
        inputs = processor(audio=self.get_audio())

        # The reserved id means "has not spoken yet" and is one past the codec's range; standing in for silence
        # with it is off-distribution and degrades generation, so the fill has to be real codes.
        codebook_size = processor.audio_tokenizer.config.codebook_size
        self.assertTrue((inputs["assistant_audio_codes"] < codebook_size).all())
        self.assertEqual(inputs["assistant_audio_codes"].shape, inputs["user_audio_codes"].shape)

    def test_no_input_raises(self):
        with self.assertRaises(ValueError):
            self.get_processor()()

    def test_save_load_pretrained(self):
        audio = self.get_audio()

        with tempfile.TemporaryDirectory() as tmpdir:
            # The processor serialises the codec by reference (class name + `name_or_path`), not by value, so the
            # codec has to be loadable from a path for the round trip to work.
            codec_dir = f"{tmpdir}/codec"
            self.get_processor().audio_tokenizer.save_pretrained(codec_dir)
            processor = self.get_processor()
            processor.audio_tokenizer = MimiModel.from_pretrained(codec_dir).eval()

            processor.save_pretrained(tmpdir)
            reloaded = AutoProcessor.from_pretrained(tmpdir)

        self.assertIsInstance(reloaded, MoshiProcessor)
        expected = processor(text="hello", audio=audio)
        actual = reloaded(text="hello", audio=audio)
        self.assertEqual(sorted(expected.keys()), sorted(actual.keys()))
        for key in expected:
            self.assertTrue((expected[key] == actual[key]).all(), f"mismatch for {key}")
