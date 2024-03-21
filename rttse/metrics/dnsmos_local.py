import os

import librosa
import numpy as np
from torch import Tensor, tensor, cat
import torchaudio
from torchmetrics import Metric
import onnxruntime as ort

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

class DNSMOSScore(Metric):
    def __init__(self, fs, primary_model_path = None, p808_model_path = None, personalized_MOS = False, **kwargs) -> None:
        super().__init__(**kwargs)
        # Get the current directory of this file
        current_dir = os.path.dirname(os.path.realpath(__file__))
        p808_model_path = p808_model_path if p808_model_path else os.path.join(current_dir, 'DNSMOS', 'model_v8.onnx')
        if personalized_MOS:
            primary_model_path = primary_model_path if primary_model_path else os.path.join(current_dir, 'pDNSMOS', 'sig_bak_ovr.onnx')
        else:
            primary_model_path = primary_model_path if primary_model_path else os.path.join(current_dir, 'DNSMOS', 'sig_bak_ovr.onnx')
        
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
        self.sampling_rate = 16000
        self.input_sampling_rate = fs
        self.is_personalized_MOS = personalized_MOS
        
        self.add_state("OVRL_raw", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("SIG_raw", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("BAK_raw", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("OVRL", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("SIG", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("BAK", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("P808_MOS", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
               
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        dim = preds.dim()
        if dim == 1:
            self.update_clip(preds)
        elif dim == 2:
            self.update_clip(preds.squeeze())
        elif dim == 3:
            for predection in preds:
                self.update_clip(predection.squeeze())
        else:
            raise ValueError("Input tensor should be 1D, 2D or 3D")
    
    def update_clip(self, predection):
        fs = self.sampling_rate
        if self.input_sampling_rate != fs:
            audio = torchaudio.transforms.Resample(orig_freq=self.input_sampling_rate, new_freq=fs)(predection)
        else:
            audio = predection

        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH*fs)
        while len(audio) < len_samples:
            audio = cat((audio, audio), dim=0)
        
        num_hops = int((len(audio) // fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        audio = audio.cpu().numpy()
        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw,self.is_personalized_MOS)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)
        self.OVRL_raw += tensor(np.mean(predicted_mos_ovr_seg_raw))
        self.SIG_raw += tensor(np.mean(predicted_mos_sig_seg_raw))
        self.BAK_raw += tensor(np.mean(predicted_mos_bak_seg_raw))
        self.OVRL += tensor(np.mean(predicted_mos_ovr_seg))
        self.SIG += tensor(np.mean(predicted_mos_sig_seg))
        self.BAK += tensor(np.mean(predicted_mos_bak_seg))
        self.P808_MOS += tensor(np.mean(predicted_p808_mos))
        self.total += 1
    
    def compute(self) -> dict:
        clip_dict = {'OVRL_raw': self.OVRL_raw, 'SIG_raw': self.SIG_raw, 'BAK_raw': self.BAK_raw, 'OVRL': self.OVRL, 'SIG': self.SIG, 'BAK': self.BAK, 'P808_MOS': self.P808_MOS}
        for key in clip_dict:
            clip_dict[key] = clip_dict[key] / self.total
        return clip_dict