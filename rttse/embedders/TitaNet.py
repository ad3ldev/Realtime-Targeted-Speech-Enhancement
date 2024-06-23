from embedders.EmbedderWrapper import EmbedderWrapper
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from torch import Tensor, zeros

class TitaNet(EmbedderWrapper):
    def __init__(self):
        super(TitaNet, self).__init__()
        self.__setup_model()
        self.embedding_dim = 192
    
    def __setup_model(self):
        self.speaker_embedder = EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large').eval()
        for param in self.speaker_embedder.parameters():
            param.requires_grad = False

    def embed(self, audio_file_path: str) -> Tensor:
        return self.speaker_embedder.get_embedding(audio_file_path)

    def embed_batch(self, audio_file_paths: list, audios: Tensor = None) -> Tensor:
        batch_size = len(audio_file_paths)
        embeddings = zeros((batch_size, self.embedding_dim))
        for i, file_path in enumerate(audio_file_paths):
            embedding = self.embed(file_path)
            embeddings[i] = embedding
        return embeddings