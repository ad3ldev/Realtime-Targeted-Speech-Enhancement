from copy import deepcopy
from torch import Tensor

import torch

from utils.logger import get_root_logger

class PFFSNWrapper(torch.nn.Module):
    def __init__(self, speech_enhancer, speaker_embedder, initial_weights=None, strict=True, *args, **kwargs) -> None:
        super(PFFSNWrapper, self).__init__(*args, **kwargs)
        self.device = self.detect_device()
        self.speech_enhancer = speech_enhancer
        self.speaker_embedder = speaker_embedder
        if initial_weights:
            self.speech_enhancer.load_state_dict(torch.load(initial_weights), strict=strict)
            self.speaker_embedder.load_state_dict(torch.load(initial_weights), strict=False)

            get_root_logger().info(f"Loaded weights from {initial_weights}")

    def detect_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def forward(self, data) -> Tensor:
        noisy_audio = data[1]
        reference_audio = data[2]

        reference_subbands = self.speaker_embedder(reference_audio)
        
        return self.speech_enhancer({"noisy": noisy_audio, "reference_subbands": reference_subbands})
    
    def __deepcopy__(self, memo):
        return PFFSNWrapper(deepcopy(self.speech_enhancer, memo), deepcopy(self.speaker_embedder, memo))