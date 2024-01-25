from functools import partial

import torch

from base_model import BaseModel
from utils.acoustics.feature import mag_phase, drop_band, stft, istft
from utils.acoustics.mask import build_complex_ideal_ratio_mask, decompress_cIRM, build_ideal_ratio_mask

def InterSubNetModel(BaseModel):
    def __init__(self, net):
        super().__init__(net)


    def setup_training(self, cfg):
        self.torch_stft = partial(stft, n_fft=cfg['acoustics'].n_fft, hop_length=cfg['acoustics'].hop_length, win_length=cfg['acoustics'].win_length)
        self.torch_istft = partial(istft, n_fft=cfg['acoustics'].n_fft, hop_length=cfg['acoustics'].hop_length, win_length=cfg['acoustics'].win_length)
        super().setup_training(cfg)

        
    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        noisy_complex = self.torch_stft(noisy)
        clean_complex = self.torch_stft(clean)

        noisy_mag, _ = mag_phase(noisy_complex)
        ground_truth_cIRM = build_complex_ideal_ratio_mask(noisy_complex, clean_complex)  # [B, F, T, 2]
        ground_truth_cIRM = drop_band(
            ground_truth_cIRM.permute(0, 3, 1, 2),  # [B, 2, F ,T]
            self.net.num_groups_in_drop_band
        ).permute(0, 2, 3, 1)

        cRM = self.net(noisy_mag.unsqueeze(1))
        cRM = cRM.permute(0, 2, 3, 1)

        loss_dict = self.calculate_loss(cRM, ground_truth_cIRM, 'train')

        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True)

        return loss_dict['train/l_total']


    def enhance(self, noisy):
        noisy_complex = self.torch_stft(noisy)

        noisy_mag, _ = mag_phase(noisy_complex)

        noisy_mag = noisy_mag.unsqueeze(1)
        cRM = self.model(noisy_mag)
        cRM = cRM.permute(0, 2, 3, 1)

        cRM = decompress_cIRM(cRM)

        enhanced_real = cRM[..., 0] * noisy_complex.real - cRM[..., 1] * noisy_complex.imag
        enhanced_imag = cRM[..., 1] * noisy_complex.real + cRM[..., 0] * noisy_complex.imag
        enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)
        enhanced = self.torch_istft(enhanced_complex, length=noisy.size(-1))
        return enhanced

    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        enhanced = self.enhance(noisy)

        clean = clean.squeeze(0)
        enhanced = enhanced.squeeze(0)

        metrics_dict = self.calculate_metrics(enhanced, clean, 'val')

        self.log_dict(metrics_dict)

    def test_step(self, batch, batch_idx):
        noisy, clean = batch
        enhanced = self.enhance(noisy)

        clean = clean.squeeze(0)
        enhanced = enhanced.squeeze(0)

        metrics_dict = self.calculate_metrics(enhanced, clean, 'test')

        self.log_dict(metrics_dict)


if __name__ == '__main__':
    InterSubNetModel(None)