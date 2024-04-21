import torch

from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr)


class WaveformerLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(WaveformerLoss, self).__init__(*args, **kwargs)
    
    def forward(self, pred, tgt):
        return -0.9 * snr(pred, tgt).mean() - 0.1 * si_snr(pred, tgt).mean()