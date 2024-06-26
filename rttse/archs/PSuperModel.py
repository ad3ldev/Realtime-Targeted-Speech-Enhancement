from copy import deepcopy
from torch import Tensor

import torch
import torch.nn as nn

from embedders.EmbedderWrapper import EmbedderWrapper

from utils.logger import get_root_logger

class PSuperModel(torch.nn.Module):
    def __init__(self, speech_enhancer, speech_embedder: EmbedderWrapper, reference_embedding, 
                 target_embedding, initial_weights=None, strict=True, *args, **kwargs) -> None:
        super(PSuperModel, self).__init__(*args, **kwargs)
        self.device = self.detect_device()
        self.speaker_embedder = speech_embedder
        self.speech_enhancer = speech_enhancer
        self.reference_embedding = reference_embedding.to(self.device)
        self.target_embedding = target_embedding.to(self.device)
        if initial_weights:
            self.speech_enhancer.load_state_dict(torch.load(initial_weights), strict=strict)
            get_root_logger().info(f"Loaded weights from {initial_weights}")

        self.speaker_embedding = nn.Sequential(
            nn.Linear(reference_embedding, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, target_embedding),
            nn.LayerNorm(target_embedding),
            nn.ReLU())

    def detect_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def forward(self, data) -> Tensor:
        noisy_audio = data["noisy"].to(self.device)
        reference_pathes = data["reference_path"]
        references = data["reference"]

        labels = self.speaker_embedder.embed_batch(reference_pathes, references).to(self.device)
        out = self.speech_enhancer(noisy_audio, self.speaker_embedding(labels))
        if type(out) == tuple:
            return out[0]
        return out
    
    def __deepcopy__(self, memo):
        return PSuperModel(deepcopy(self.speech_enhancer, memo), self.speaker_embedder, self.reference_embedding, self.target_embedding)