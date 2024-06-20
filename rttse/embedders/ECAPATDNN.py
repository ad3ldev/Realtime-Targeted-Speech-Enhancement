from embedders.EmbedderWrapper import EmbedderWrapper
from speechbrain.inference.speaker import EncoderClassifier
from torch import Tensor, cuda
import torchaudio

from utils.logger import get_root_logger

class ECAPATDNN(EmbedderWrapper):
    def __init__(self):
        super(ECAPATDNN, self).__init__()
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.model = EncoderClassifier(run_opts={'device': self.device}).from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    def embed(self, file_path: str) -> Tensor: 
        signal, _ = torchaudio.load(file_path)
        get_root_logger().info(f'ECAPA-TDNN signal tensor device: {signal.device}')
        get_root_logger().info(f'ECAPA-TDNN model device: {self.model.device}')
        return self.model.encode_batch(signal).squeeze(0)