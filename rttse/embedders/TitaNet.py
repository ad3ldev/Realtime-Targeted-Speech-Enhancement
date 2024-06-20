from embedders.EmbedderWrapper import EmbedderWrapper
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from torch import Tensor, zeros

class TitaNet(EmbedderWrapper):
    def __init__(self):
        super(TitaNet, self).__init__()
        self.speaker_embedder = EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')

    def embed(self, audio_file_path: str) -> Tensor:
        return self.speaker_embedder.get_embedding(audio_file_path)

    def embed_batch(self, audio_file_paths: list, audios: Tensor = None) -> Tensor:
        embeddings = zeros((len(audio_file_paths), 192))
        for i, file_path in enumerate(audio_file_paths):
            embedding = self.embed(file_path)
            embeddings[i] = embedding
        return embeddings