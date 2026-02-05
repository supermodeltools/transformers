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
import io

from ...utils import auto_docstring, is_mistral_common_available, is_soundfile_available, is_torch_available, logging


if is_torch_available():
    import torch

if is_soundfile_available():
    import soundfile as sf

if is_mistral_common_available():
    from mistral_common.audio import Audio
    from mistral_common.protocol.instruct.chunk import RawAudio
    from mistral_common.protocol.transcription.request import TranscriptionRequest, StreamingMode

from ...audio_utils import AudioInput, load_audio_as, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import AllKwargsForChatTemplate, AudioKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput


logger = logging.get_logger(__name__)


class VoxtralRealtimeAudioKwargs(AudioKwargs, total=False):
    """
    is_first_iteration (`bool`, *optional*):
        Whether this is the first iteration of processing.
    """

    is_first_iteration: bool | None


class VoxtralRealtimeProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: VoxtralRealtimeAudioKwargs
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "add_special_tokens": False,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "truncation": False,
            "is_first_iteration": True,
        },
    }


@auto_docstring
class VoxtralRealtimeProcessor(ProcessorMixin):
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def __call__(
        self,
        audio: AudioInput | None = None,
        **kwargs: Unpack[VoxtralRealtimeProcessorKwargs],
    ):
        output_kwargs = self._merge_kwargs(VoxtralRealtimeProcessorKwargs, **kwargs)

        audio = make_list_of_audio(audio)
        input_ids, texts, audio_arrays = [], [], []
        for audio_el in audio:
            audio = Audio(audio_array=audio_el, sampling_rate=output_kwargs["audio_kwargs"]["sampling_rate"], format="ogg")
            transcription_request = TranscriptionRequest(
                audio=RawAudio.from_audio(audio),
                streaming=StreamingMode.OFFLINE,
                language=None,
            )
            tokenized_transcription_request = self.tokenizer.tokenizer.encode_transcription(transcription_request)

            input_ids.append(tokenized_transcription_request.tokens)
            texts.append(tokenized_transcription_request.text)
            audio_arrays.extend([el.audio_array for el in tokenized_transcription_request.audios])

        text_encoding = self.tokenizer(input_ids, **output_kwargs["text_kwargs"])
        
        is_first_iteration = output_kwargs["audio_kwargs"].pop("is_first_iteration")
        audio_encoding = self.feature_extractor(
            audio_arrays,
            center=is_first_iteration,
            **output_kwargs["audio_kwargs"],
        )
        encoding = {**text_encoding, **audio_encoding}

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return BatchFeature(data=encoding, tensor_type=return_tensors)


__all__ = ["VoxtralRealtimeProcessor"]
