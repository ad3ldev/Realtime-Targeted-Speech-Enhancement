from torch import nn
from torch.nn import functional as F

def loss_fn(net, X, ell_p, ell_p_lambda, stft_lambda, mrstftloss, **kwargs):
    """
    Loss function in CleanUNet

    Parameters:
    net: network
    X: training data pair (clean audio, noisy_audio)
    ell_p: \ell_p norm (1 or 2) of the AE loss
    ell_p_lambda: factor of the AE loss
    stft_lambda: factor of the STFT loss
    mrstftloss: multi-resolution STFT loss function

    Returns:
    loss: value of objective function
    output_dic: values of each component of loss
    """

    assert type(X) == tuple and len(X) == 2
    
    clean_audio, noisy_audio = X
    B, C, L = clean_audio.shape
    output_dic = {}
    loss = 0.0
    
    # AE loss
    denoised_audio = net(noisy_audio)  

    if ell_p == 2:
        ae_loss = nn.MSELoss()(denoised_audio, clean_audio)
    elif ell_p == 1:
        ae_loss = F.l1_loss(denoised_audio, clean_audio)
    else:
        raise NotImplementedError
    loss += ae_loss * ell_p_lambda
    output_dic["reconstruct"] = ae_loss.data * ell_p_lambda

    if stft_lambda > 0:
        sc_loss, mag_loss = mrstftloss(denoised_audio.squeeze(1), clean_audio.squeeze(1))
        loss += (sc_loss + mag_loss) * stft_lambda
        output_dic["stft_sc"] = sc_loss.data * stft_lambda
        output_dic["stft_mag"] = mag_loss.data * stft_lambda

    return loss, output_dic

class CleanUNetSTFTLoss(nn.Module):
    def __init__(self, stft_lambda, mrstftloss):
        super(CleanUNetSTFTLoss, self).__init__()
        self.stft_lambda = stft_lambda
        self.mrstftloss = mrstftloss

    def forward(self, denoised_audio, clean_audio):
        sc_loss, mag_loss = self.mrstftloss(denoised_audio.squeeze(1), clean_audio.squeeze(1))
        return (sc_loss + mag_loss) * self.stft_lambda