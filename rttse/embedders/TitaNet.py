from embedders.EmbedderWrapper import EmbedderWrapper
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from torch import Tensor

class TitaNet(EmbedderWrapper):
    def __init__(self):
        super(TitaNet, self).__init__()
        self.speaker_embedder = EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')

    def embed(self, file_path: str) -> Tensor:
        return self.speaker_embedder.get_embedding(file_path)