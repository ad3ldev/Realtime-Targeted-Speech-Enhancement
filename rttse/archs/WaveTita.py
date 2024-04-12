from copy import deepcopy
from torch import Tensor

import torch

from nemo.collections.asr.models import EncDecSpeakerLabelModel

from utils.logger import get_root_logger

class WaveTita(torch.nn.Module):
    def __init__(self, speech_enhancer, initial_weights=None, *args, **kwargs) -> None:
        super(WaveTita, self).__init__(*args, **kwargs)
        self.device = self.detect_device()
        self.speaker_embedder = self.load_speaker_embedder()
        self.speech_enhancer = speech_enhancer
        if initial_weights:
            self.speech_enhancer.load_state_dict(torch.load(initial_weights), strict=kwargs.get("strict", True))
            get_root_logger().info(f"Loaded weights from {initial_weights}")

    def detect_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_speaker_embedder(self) -> None:
        speaker_embedder = EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large').eval()
        # Freeze the speaker embedder
        for param in speaker_embedder.parameters():
            param.requires_grad = False
        return speaker_embedder
    
    def forward(self, data) -> Tensor:
        noisy_audio = data["noisy"].to(self.device)
        reference_path = data["reference_path"]
        batch_size = noisy_audio.shape[0]

        labels = torch.zeros((batch_size, 192))
        for j in range(batch_size):
            labels[j] = self.speaker_embedder.get_embedding(reference_path[j])

        labels = labels.to(self.device)
        return self.speech_enhancer(noisy_audio, labels)
    
    def __deepcopy__(self, memo):
        return WaveTita(deepcopy(self.speech_enhancer, memo))