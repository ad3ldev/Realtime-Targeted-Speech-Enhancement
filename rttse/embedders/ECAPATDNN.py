from rttse.embedders.EmbedderWrapper import EmbedderWrapper
from speechbrain.inference.speaker import EncoderClassifier
from torch import Tensor
import torchaudio

class ECAPATDNN(EmbedderWrapper):
    def __init__(self):
        self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    def embed(self, file_path: str) -> Tensor: 
        signal, _ = torchaudio.load(file_path)
        return self.model.encode_batch(signal).squeeze(0)