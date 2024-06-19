from copy import deepcopy
from torch import Tensor

import torch

from embedders.EmbedderWrapper import EmbedderWrapper

from utils.logger import get_root_logger

class WaveTita(torch.nn.Module):
    def __init__(self, speech_enhancer, speech_embedder: EmbedderWrapper, initial_weights=None, strict=True, *args, **kwargs) -> None:
        super(WaveTita, self).__init__(*args, **kwargs)
        self.device = self.detect_device()
        self.speaker_embedder = speech_embedder
        self.speech_enhancer = speech_enhancer
        if initial_weights:
            self.speech_enhancer.load_state_dict(torch.load(initial_weights), strict=strict)
            get_root_logger().info(f"Loaded weights from {initial_weights}")

    def detect_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def forward(self, data) -> Tensor:
        noisy_audio = data["noisy"].to(self.device)
        reference_pathes = data["reference_path"]

        labels = self.speaker_embedder.embed_batch(reference_pathes).to(self.device)

        return self.speech_enhancer(noisy_audio, labels)
    
    def __deepcopy__(self, memo):
        return WaveTita(deepcopy(self.speech_enhancer, memo), self.speaker_embedder)