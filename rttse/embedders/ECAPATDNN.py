from embedders.EmbedderWrapper import EmbedderWrapper
from speechbrain.inference.speaker import EncoderClassifier
from torch import Tensor, cuda
import torchaudio

class ECAPATDNN(EmbedderWrapper):
    def __init__(self):
        super(ECAPATDNN, self).__init__()
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    def embed(self, file_path: str) -> Tensor: 
        signal, _ = torchaudio.load(file_path)
        signal = signal.to(self.device)
        return self.model.encode_batch(signal).squeeze(0)