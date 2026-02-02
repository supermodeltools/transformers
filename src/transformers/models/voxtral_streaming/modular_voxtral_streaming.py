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
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
from ...models.mistral.modeling_mistral import MistralMLP, MistralRMSNorm
from ...models.voxtral.modeling_voxtral import (
    VoxtralForConditionalGeneration,
    VoxtralPreTrainedModel,
)
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs
from .configuration_voxtral_streaming import VoxtralStreamingEncoderConfig


class Conv1dCacheLayer:
    def __init__(self, conv_config):
        self.in_channels = conv_config["in_channels"]
        self.left_pad = (conv_config["kernel_size"] - 1) * conv_config["dilation"] + 1 - conv_config["stride"]
        self.cache: torch.Tensor | None = None
        self.is_initialized: bool = False

    def update(self, hidden_states):
        batch_size = hidden_states.shape[0]

        if not self.is_initialized:
            self.cache = torch.zeros(batch_size, self.in_channels, self.left_pad, device=hidden_states.device, dtype=hidden_states.dtype)
            self.output_cache = torch.zeros_like(self.cache)
            self.is_initialized = True

        # double buffer to keep tensors static for compile
        self.output_cache, self.cache = self.cache, self.output_cache

        # get the padding states
        if self.left_pad > 0:
            shortfall = max(0, self.left_pad - hidden_states.shape[-1])
            if shortfall > 0:
                padding_states = torch.cat([self.output_cache[:, :, -shortfall:], hidden_states], dim=-1)
            else:
                padding_states = hidden_states[:, :, -self.left_pad :]
        else:
            padding_states = torch.empty(
                batch_size, self.in_channels, 0, dtype=hidden_states.dtype, device=hidden_states.device
            )

        # update the cache
        self.cache.copy_(padding_states)

        return self.output_cache


class VoxtralStreamingConv1dPaddingCache:
    def __init__(self, config):
        if not hasattr(config, "_conv_config"):
            raise ValueError("TODO")

        self.layers = [Conv1dCacheLayer(conv_config) for conv_config in config._conv_config]

    def update(self, hidden_states, layer_idx):
        padding_states = self.layers[layer_idx].update(hidden_states)
        padded_hidden_states = torch.cat([padding_states, hidden_states], dim=-1)
        return padded_hidden_states


@dataclass
class VoxtralStreamingEncoderOutput(BaseModelOutputWithPast):
    padding_cache: VoxtralStreamingConv1dPaddingCache | None = None


class VoxtralStreamingRotaryEmbedding(LlamaRotaryEmbedding): ...


class VoxtralStreamingCausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        conv_layer_idx: int | None = 0,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        self._stride = self.stride[0]
        self._effective_kernel_size = (kernel_size - 1) * self.dilation[0] + 1
        self._padding_total = self._effective_kernel_size - self._stride
        self.conv_layer_idx = conv_layer_idx

    def forward(self, x: torch.Tensor, padding_cache: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        if padding_cache is not None:
            x = padding_cache.update(x, self.conv_layer_idx)

        x = super().forward(x)

        if mask is not None:
            mask = nn.functional.pad(mask, (self.left_pad, 0))[:, None, :]
            weight = torch.ones(1, 1, self.kernel_size[0], device=mask.device)
            mask = nn.functional.conv1d(mask.float(), weight, stride=self.stride)
            mask = mask > 0
            x *= mask

        if mask is not None:
            mask = mask.squeeze(1)
        return x


class VoxtralStreamingRMSNorm(MistralRMSNorm): ...


class VoxtralStreamingAttention(LlamaAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)


class VoxtralStreamingMLP(MistralMLP):
    def __init__(self, config):
        super().__init__(config)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)


class VoxtralStreamingEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv1 = VoxtralStreamingCausalConv1d(config.num_mel_bins, config.d_model, kernel_size=3, conv_layer_idx=0)
        self.conv2 = VoxtralStreamingCausalConv1d(config.d_model, config.d_model, kernel_size=3, stride=2, conv_layer_idx=1)

    def forward(self, input_features, padding_cache=None):
        inputs_embeds = nn.functional.gelu(self.conv1(input_features, padding_cache=padding_cache))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds, padding_cache=padding_cache))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        return inputs_embeds


class VoxtralStreamingEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = VoxtralStreamingAttention(config, layer_idx)
        self.self_attn_layer_norm = VoxtralStreamingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.final_layer_norm = VoxtralStreamingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = VoxtralStreamingMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class VoxtralStreamingPreTrainedModel(VoxtralPreTrainedModel, PreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(module)
        if isinstance(module, TimeEmbedding):
            inv_freq = torch.exp(-math.log(module.theta) * torch.arange(module.dim // 2).float() / (module.dim // 2))
            init.copy_(module.inv_freq, inv_freq)


@auto_docstring(
    custom_intro="""
    The VoxtralStreaming encoder, which is a Whisper encoder.
    """
)
class VoxtralStreamingEncoder(VoxtralStreamingPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`VoxtralStreamingEncoderLayer`].

    Args:
        config: VoxtralStreamingEncoderConfig
    """

    # Ignore copy
    config: VoxtralStreamingEncoderConfig
    main_input_name = "input_features"
    input_modalities = "audio"
    _no_split_modules = ["VoxtralStreamingEncoderLayer"]
    _can_record_outputs = {
        "attentions": VoxtralStreamingAttention,
        "hidden_states": VoxtralStreamingEncoderLayer,
    }

    def __init__(self, config):
        super().__init__(config)
        self.embedder = VoxtralStreamingEmbedder(config)
        self.layers = nn.ModuleList(
            [VoxtralStreamingEncoderLayer(config, layer_idx) for layer_idx in range(config.encoder_layers)]
        )
        self.norm = VoxtralStreamingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = VoxtralStreamingRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_features=None,
        padding_cache=None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        padding_mask: torch.Tensor | None = None,
        inputs_embeds=None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        use_padding_cache: bool | None = None,
        attention_mask=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        if (input_features is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_features or inputs_embeds")

        if use_padding_cache and padding_cache is None:
            padding_cache = VoxtralStreamingConv1dPaddingCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embedder(input_features, padding_cache)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return VoxtralStreamingEncoderOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            padding_cache=padding_cache,
        )


# class MistralStreamingAdaRmsNorm(nn.Module):
#     def __init__(self, config: MistralConfig):
#         super().__init__()
#         # TODO: how to add the intermediate size to the config? since it already the mistral one? new model? new config only?
#         self.linear1 = nn.Linear(config.hidden_size, 32, bias=False)
#         self.linear2 = nn.Linear(32, config.hidden_size, bias=False)

#     def forward(self, hidden_states):
#         hidden_states = self.linear1(hidden_states)
#         hidden_states = nn.functional.gelu(hidden_states)
#         hidden_states = self.linear2(hidden_states)
#         return hidden_states


# class MistralStreamingDecoderLayer(MistralDecoderLayer):
#     def __init__(self, config: MistralConfig, layer_idx: int):
#         super().__init__(config, layer_idx)
#         self.ada_rms_norm = MistralStreamingAdaRmsNorm(config)


# class MistralStreamingForCausalLM(MistralForCausalLM): ...


class TimeEmbedding(nn.Module):
    """Sinusoidal Embedding for encoding time"""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = torch.exp(-math.log(self.theta) * torch.arange(self.dim // 2).float() / (self.dim // 2))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t[..., None]  # (B,) -> (B, 1) or (B, T) -> (B, T, 1)
        inv_freq = self.inv_freq.to(device=t.device, dtype=t.dtype)
        emb = t * inv_freq  # (B, 1) x (D/2,) -> (B, D/2) or (B, T, 1) x (D/2,) -> (B, T, D/2)
        return torch.cat((emb.cos(), emb.sin()), dim=-1)  # (B, D) or (B, T, D)


class VoxtralStreamingForConditionalGeneration(VoxtralForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.time_embedding = TimeEmbedding(config.text_config.hidden_size)

    @can_return_tuple
    @auto_docstring(
        custom_intro="This method is used to get the audio embeddings from input features (a log mel spectrogram), meaning inferring the audio encoder and the multi-modal projector."
    )
    def get_audio_features(
        self,
        input_features: torch.FloatTensor = None,
        padding_cache: torch.FloatTensor | None = None,
        encoder_inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features (`torch.FloatTensor`):
            Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
            obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]` or a
            `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
            `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
            and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        """
        audio_outputs = self.audio_tower(
            input_features=input_features,
            inputs_embeds=encoder_inputs_embeds,
            past_key_values=past_key_values,
            padding_cache=padding_cache,
            return_dict=True,
            use_cache=True,
            use_padding_cache=True,
            **kwargs,
        )
        audio_hidden_states = audio_outputs.last_hidden_state
        audio_hidden_states = audio_hidden_states.reshape(-1, self.config.audio_config.intermediate_size)
        audio_embeds = self.multi_modal_projector(audio_hidden_states)
        audio_outputs.pooler_output = audio_embeds

        return audio_outputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        encoder_past_key_values: Cache | None = None,
        padding_cache: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import VoxtralStreamingForConditionalGeneration, AutoProcessor
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> repo_id = "mistralai/VoxtralStreaming-Mini-3B-2507"

        >>> processor = AutoProcessor.from_pretrained(repo_id)
        >>> model = VoxtralStreamingForConditionalGeneration.from_pretrained(repo_id, dtype=torch.bfloat16, device_map=device)

        >>> conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dude_where_is_my_car.wav",
                    },
                    {"type": "text", "text": "What can you tell me about this audio?"},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(conversation)
        >>> inputs = inputs.to(device, dtype=torch.bfloat16)

        >>> outputs = model.generate(**inputs, max_new_tokens=30)
        >>> processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        ["This audio is a humorous conversation between two friends, likely in English, where one of them is trying to figure out what the other's tattoo says."]
        ```"""
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_features is not None or encoder_inputs_embeds is not None:
            audio_outputs = self.get_audio_features(
                input_features=input_features,
                encoder_inputs_embeds=encoder_inputs_embeds,
                past_key_values=encoder_past_key_values,
                padding_cache=padding_cache,
                return_dict=True,
            )
            inputs_embeds += audio_outputs.pooler_output

        time_tensor = torch.full(
            (1,),
            fill_value=6,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        t_cond = self.time_embedding(time_tensor)

        outputs: BaseModelOutputWithPast = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            t_cond=t_cond,
            **kwargs,
        )
        outputs["encoder_past_key_values"] = audio_outputs.past_key_values
        outputs["padding_cache"] = audio_outputs.padding_cache
        return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # Overwritten -- we should not pass input_features when we are in cached decoding stage

        input_features = kwargs.pop("input_features", None)
        # is_first_iteration = kwargs.get("is_first_iteration", False)

        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        # if is_first_iteration or not kwargs.get("use_cache", True):
        #     # input_features should only be passed when we are not in cached decoding stage
        #     model_inputs["input_features"] = input_features
        #     # model_inputs["input_features"] = model_inputs["input_features"][..., start_idx:end_idx]
        #     self.encoder_inputs_embeds = self.audio_tower.embedder(input_features)

        start_idx = model_inputs["cache_position"][0] * 4
        end_idx = (model_inputs["cache_position"][-1] + 1) * 4

        # model_inputs["encoder_inputs_embeds"] = self.encoder_inputs_embeds[:, start_idx:end_idx, :]
        # model_inputs.pop("input_features", None)

        start_idx *= 2
        end_idx *= 2
        model_inputs["input_features"] = input_features[..., start_idx:end_idx]

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )

        if hasattr(outputs, "encoder_past_key_values"):
            model_kwargs["encoder_past_key_values"] = outputs.encoder_past_key_values

        if hasattr(outputs, "padding_cache"):
            model_kwargs["padding_cache"] = outputs.padding_cache

        return model_kwargs


__all__ = [
    "VoxtralStreamingForConditionalGeneration",
    "VoxtralStreamingEncoder",
]
