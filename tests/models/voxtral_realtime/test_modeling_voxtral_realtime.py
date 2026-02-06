# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch VoxtralRealtime model."""

import tempfile
import unittest
import functools

from transformers import (
    VoxtralRealtimeConfig,
    VoxtralRealtimeForConditionalGeneration,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch


class VoxtralRealtimeModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        audio_token_id=0,
        seq_length=5,
        feat_seq_length=40,
        text_config={
            "model_type": "voxtral_realtime_text",
            "intermediate_size": 36,
            "initializer_range": 0.02,
            "hidden_size": 32,
            "max_position_embeddings": 52,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "use_labels": True,
            "vocab_size": 99,
            "head_dim": 8,
            "pad_token_id": 1,  # can't be the same as the audio token id
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "attention_dropout": 0.0,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
        },
        is_training=True,
        audio_config={
            "model_type": "voxtral_realtime_encoder",
            "hidden_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 64,
            "encoder_layers": 2,
            "num_mel_bins": 80,
            "max_position_embeddings": 100,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "activation_function": "silu",
            "activation_dropout": 0.0,
            "attention_dropout": 0.0,
            "head_dim": 4,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.audio_token_id = audio_token_id
        self.text_config = text_config
        self.audio_config = audio_config
        self.seq_length = seq_length
        self.feat_seq_length = feat_seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.encoder_seq_length = seq_length
        self._max_new_tokens = None  # this is used to set

    def get_config(self):
        return VoxtralRealtimeConfig(
            text_config=self.text_config,
            audio_config=self.audio_config,
            ignore_index=self.ignore_index,
            audio_token_id=self.audio_token_id,
        )

    def prepare_config_and_inputs(self):
        if self._max_new_tokens is not None:
            feat_seq_length = self.feat_seq_length + self._max_new_tokens * 8
        else:
            feat_seq_length = self.feat_seq_length

        input_features_values = floats_tensor(
            [
                self.batch_size,
                self.audio_config["num_mel_bins"],
                feat_seq_length,
            ]
        )
        config = self.get_config()
        return config, input_features_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_features_values = config_and_inputs
        num_audio_tokens_per_batch_idx = 30

        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)
        attention_mask[:, :1] = 0

        input_ids[:, 1 : 1 + num_audio_tokens_per_batch_idx] = config.audio_token_id
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features_values,
        }
        return config, inputs_dict


@require_torch
class VoxtralRealtimeForConditionalGenerationModelTest(
    ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    """
    Model tester for `VoxtralRealtimeForConditionalGeneration`.
    """
    additional_model_inputs = ["input_features"]

    all_model_classes = (VoxtralRealtimeForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"any-to-any": VoxtralRealtimeForConditionalGeneration}
        if is_torch_available()
        else {}
    )

    _is_composite = True

    def setUp(self):
        self.model_tester = VoxtralRealtimeModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VoxtralRealtimeConfig, has_text_modality=False)

    def _with_max_new_tokens(max_new_tokens):
        def decorator(test_func):
            @functools.wraps(test_func)
            def wrapper(self, *args, **kwargs):
                try:
                    self.model_tester._max_new_tokens = max_new_tokens
                    return test_func(self, *args, **kwargs)
                finally:
                    self.model_tester._max_new_tokens = None
            return wrapper
        return decorator

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        original_feat_seq_length = self.model_tester.feat_seq_length
        try:
            self.model_tester.feat_seq_length += self.max_new_tokens * 8
            config, inputs_dict = super().prepare_config_and_inputs_for_generate(batch_size=batch_size)
        finally:
            self.model_tester.feat_seq_length = original_feat_seq_length
        return config, inputs_dict

    @_with_max_new_tokens(max_new_tokens=10)
    def test_generate_methods_with_logits_to_keep(self):
        super().test_generate_methods_with_logits_to_keep()

    @_with_max_new_tokens(max_new_tokens=5)
    def test_generate_compile_model_forward_fullgraph(self):
        super().test_generate_compile_model_forward_fullgraph()
    
    @_with_max_new_tokens(max_new_tokens=4)
    def test_generate_continue_from_past_key_values(self):
        super().test_generate_continue_from_past_key_values()

    @unittest.skip(
        reason="VoxtralRealtime does not have a base model"
    )
    def test_model_base_model_prefix(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime since input_features must be provided along input_ids"
    )
    def test_flash_attention_2_continue_generate_with_position_ids(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime since input_features must be provided along input_ids"
    )
    def test_custom_4d_attention_mask(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime since input_features must be provided along input_ids"
    )
    def test_flash_attn_2_from_config(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime since input_features must be provided along input_ids"
    )
    def attention_mask_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime since input_features must be provided along input_ids"
    )
    def flash_attn_inference_equivalence(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime for now since encoder_past_key_values AND padding_cache are returned by generate"
    )
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime since prepare_inputs_for_generation is overwritten"
    )
    def test_prepare_inputs_for_generation_kwargs_forwards(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime since input_features must be provided along input_ids"
    )
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(
        reason="VoxtralRealtime does not fall in the paradigm of assisted decoding (at least for the way it is implemented in generate)"
    )
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip(
        reason="VoxtralRealtime does not fall in the paradigm of assisted decoding (at least for the way it is implemented in generate)"
    )
    def test_assisted_decoding_matches_greedy_search_0_random(self):
        pass

    @unittest.skip(
        reason="VoxtralRealtime does not fall in the paradigm of assisted decoding (at least for the way it is implemented in generate)"
    )
    def test_assisted_decoding_matches_greedy_search_1_same(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime since in only pads input_ids but input_features should also be padded"
    )
    def test_left_padding_compatibility(self):
        pass


# TODO: Add integration tests once checkpoint is available
# @require_torch
# class VoxtralRealtimeForConditionalGenerationIntegrationTest(unittest.TestCase):
#     def setUp(self):
#         self.checkpoint_name = "mistralai/VoxtralRealtime-Mini-3B-2507"
#         self.dtype = torch.bfloat16
#         self.processor = AutoProcessor.from_pretrained(self.checkpoint_name)
#
#     def tearDown(self):
#         cleanup(torch_device, gc_collect=True)
#
#     @slow
#     def test_realtime_streaming_inference(self):
#         """Test streaming inference with the realtime model."""
#         pass
