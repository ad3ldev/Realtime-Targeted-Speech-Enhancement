from embedders.EmbedderWrapper import EmbedderWrapper
from speechbrain.inference.speaker import EncoderClassifier
from torch import Tensor, cuda
import torchaudio

class ECAPATDNN(EmbedderWrapper):
    def __init__(self, device = None):
        super(ECAPATDNN, self).__init__()
        self.device = device if device is not None else "cuda" if cuda.is_available() else "cpu"
        self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": self.device})
        self.model.hparams.label_encoder.ignore_len()
        self.sr = 16000

    def embed(self, audio_file_path: str) -> Tensor: 
        signal, signal_sr = torchaudio.load(audio_file_path)
        signal = signal.to(self.device)
        if signal_sr != self.sr:
            signal = torchaudio.transforms.Resample(signal_sr, self.sr)(signal)
        return self.model.encode_batch(signal).squeeze(1)

    def embed_batch(self, audio_file_paths: list, audios: Tensor = None) -> Tensor:
        if audios is None:
            raise ValueError("Audios tensor is None!")

        return self.model.encode_batch(audios).squeeze(1)
