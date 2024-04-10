import os

import torch
import torchaudio
from torch import tensor
from torch.nn import Module
from torchmetrics import Metric

import onnx
import onnx2torch

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

def polyval(x,coeffs):
    curVal=0
    for curValIndex in range(len(coeffs)-1):
        curVal=(curVal+coeffs[curValIndex])*x
    return(curVal+coeffs[len(coeffs)-1])

def power_to_db(S, ref = tensor(1.0), amin = tensor(1e-10), top_db = tensor(80.0)):
    magnitude = S
    if callable(ref):
        ref_value = ref(magnitude)
    else:
        ref_value = torch.abs(ref)
    log_spec = 10.0 * torch.log10(torch.maximum(amin, magnitude))
    log_spec -= 10.0 * torch.log10(torch.maximum(amin, ref_value))
    if top_db is not None:
        log_spec = torch.maximum(log_spec, log_spec.max() - top_db)
    return log_spec

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
        
        primary_model_onnx = onnx.load(primary_model_path)
        p808_model_onnx = onnx.load(p808_model_path)
        
        self.primary_model = onnx2torch.convert(primary_model_onnx).to(self.device)
        self.p808_model = onnx2torch.convert(p808_model_onnx).to(self.device)
        
        self.primary_model.requires_grad_(False)
        self.p808_model.requires_grad_(False)
        
        self.sampling_rate = SAMPLING_RATE
        self.input_sampling_rate = fs
        
        if self.sampling_rate != self.input_sampling_rate:            
            self.resmapler = torchaudio.transforms.Resample(orig_freq=self.input_sampling_rate, new_freq=self.sampling_rate).to(self.device)
        else:
            self.resmapler = None
                    
        self.is_personalized_MOS = personalized_MOS
        
        self.add_state("OVRL_raw", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("SIG_raw", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("BAK_raw", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("OVRL", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("SIG", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("BAK", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("P808_MOS", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
    
    def audio_melspec(self, audio: tensor, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels, norm='slaney', mel_scale='slaney').to(self.device)
        mel_spec = transform(audio)
        if to_db:
            mel_spec = (power_to_db(mel_spec, ref=torch.max)+40)/40
        return mel_spec.T
    
    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = torch.tensor([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = torch.tensor([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = torch.tensor([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = torch.tensor([-0.06766283,  1.11546468,  0.04602535])
            p_sig = torch.tensor([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = torch.tensor([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = polyval(sig, p_sig)
        bak_poly = polyval(bak, p_bak)
        ovr_poly = polyval(ovr, p_ovr)
        
        return sig_poly, bak_poly, ovr_poly
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
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
            audio = self.resmapler(predection)
        else:
            audio = predection

        len_samples = int(INPUT_LENGTH*fs)
        while len(audio) < len_samples:
            audio = torch.cat((audio, audio), dim=0)
        
        num_hops = int((len(audio) // fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        # audio = audio.cpu().numpy()
        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue
            
            input_features = audio_seg.unsqueeze(0)
            p808_input_features = self.audio_melspec(audio=audio_seg[:-160]).unsqueeze(0)
            # with torch.no_grad():
            p808_mos = self.p808_model(p808_input_features)[0][0]
            primary_model_result = self.primary_model(input_features)[0]
            mos_sig_raw,mos_bak_raw,mos_ovr_raw = primary_model_result[0], primary_model_result[1], primary_model_result[2]
            mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw,self.is_personalized_MOS)

            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)
            
        self.OVRL_raw += torch.mean(tensor(predicted_mos_ovr_seg_raw))
        self.SIG_raw += torch.mean(tensor(predicted_mos_sig_seg_raw))
        self.BAK_raw += torch.mean(tensor(predicted_mos_bak_seg_raw))
        self.OVRL += torch.mean(tensor(predicted_mos_ovr_seg))
        self.SIG += torch.mean(tensor(predicted_mos_sig_seg))
        self.BAK += torch.mean(tensor(predicted_mos_bak_seg))
        self.P808_MOS += torch.mean(tensor(predicted_p808_mos))
        self.total += 1
    
    def compute(self) -> dict:
        clip_dict = {'OVRL_raw': self.OVRL_raw, 'SIG_raw': self.SIG_raw, 'BAK_raw': self.BAK_raw, 'OVRL': self.OVRL, 'SIG': self.SIG, 'BAK': self.BAK, 'P808_MOS': self.P808_MOS}
        for key in clip_dict:
            clip_dict[key] = clip_dict[key] / self.total
        return clip_dict
    
    def to(self, device):
        this = super().to(device)
        this.primary_model.to(device)
        this.p808_model.to(device)
        if this.resmapler:
            this.resmapler.to(device)
        return this
    
    def _apply(self, fn , exclude_state = "") -> Module:
        this = super()._apply(fn, exclude_state)
        self.primary_model.to(self.device)
        self.p808_model.to(self.device)
        if self.resmapler:
            self.resmapler.to(self.device)
        return this
    
# import argparse
# from tqdm import tqdm
# def main():
#     # Source files path
#     parser = argparse.ArgumentParser(description='DNSMOS')
#     parser.add_argument('--primary_model_path', type=str, default=None, help='Primary model path')
#     parser.add_argument('--p808_model_path', type=str, default=None, help='P808 model path')
#     parser.add_argument('--personalized_MOS', action='store_true', help='Use personalized MOS')
#     parser.add_argument('-t', "--testset_dir", default='.', 
#                         help='Path to the dir containing audio clips in .wav to be evaluated')
#     parser.add_argument('-fs', "--sampling_rate", default=16000, help='Sampling rate of the audio clips')
#     # parser.add_argument('-o', "--csv_path", default=None, help='Dir to the csv that saves the results')
#     args = parser.parse_args()
    
#     # Get the files in the testset directory
#     testset_dir = args.testset_dir
#     files = [f for f in os.listdir(testset_dir) if f.endswith('.wav')]
#     # Initialize the metric
#     dns_mos = DNSMOSScore(fs=args.sampling_rate, primary_model_path=args.primary_model_path, p808_model_path=args.p808_model_path, personalized_MOS=args.personalized_MOS).to('cuda')
#     # Evaluate the audio clips
    
#     rows = []
    
#     for file in tqdm(files):
#         audio, sr = torchaudio.load(os.path.join(testset_dir, file))
#         if torch.cuda.is_available():
#             audio = audio.cuda()
#         dns_mos(audio, tensor(0.0))
    
#     # Compute the metric
#     results = dns_mos.compute()
#     print(results)

# if __name__ == '__main__':
#     main()