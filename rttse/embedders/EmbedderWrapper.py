from torch import Tensor, zeros, nn
from utils.logger import get_root_logger

class EmbedderWrapper(nn.Module):
    def __init__(self):
        super(EmbedderWrapper, self).__init__()

    def embed(self, audio_file_path: str) -> Tensor:
        pass

    def embed_batch(self, audio_file_paths: list, audios: Tensor = None) -> Tensor:
        pass
    
    def __call__(self, audio_file_path: str) -> Tensor:
        return self.embed(audio_file_path)