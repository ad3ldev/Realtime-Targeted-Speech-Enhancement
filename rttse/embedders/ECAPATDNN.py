from embedders.EmbedderWrapper import EmbedderWrapper
from speechbrain.inference.speaker import EncoderClassifier
from torch import Tensor, cuda, cat
import torchaudio

from utils.logger import get_root_logger

class ECAPATDNN(EmbedderWrapper):
    def __init__(self):
        super(ECAPATDNN, self).__init__()
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": self.device})

    def embed(self, file_path: str) -> Tensor: 
        signal, _ = torchaudio.load(file_path)
        signal = signal.to(self.device)
        get_root_logger().info(f'ECAPA-TDNN signal tensor shape: {signal.shape}')
        get_root_logger().info(f'ECAPA-TDNN signal tensor device: {signal.device}')
        get_root_logger().info(f'ECAPA-TDNN model device: {self.model.device}')
        return self.model.encode_batch(signal).squeeze(1)

    def embed_batch(self, file_paths: list) -> Tensor:
        audios_tensors = []
        min_length = float('inf')

        for file_path in file_paths:
            signal = torchaudio.load(file_path)[0].to(self.device)
            min_length = min(min_length, signal.size(1))
            audios_tensors.append(signal)

        for i in range(len(audios_tensors)):
            audios_tensors[i] = audios_tensors[i][:, :min_length]

        return self.model.encode_batch(cat(audios_tensors)).squeeze(1)
