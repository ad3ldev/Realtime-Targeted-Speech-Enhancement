import torch
import torch.nn as nn
import torchaudio as audio
from torch.nn import functional as F

from utils.fastfullsubnetutils import stft
from archs.FastFullSubNet import FastFullSubNet


class PersonalizedFastFullSubnet(FastFullSubNet):
    def __init__(
        self,
        look_ahead=2,
        shrink_size=2,
        sequence_model="LSTM",
        num_mels=64,
        encoder_input_size=257,
        bottleneck_hidden_size=384,
        bottleneck_num_layers=2,
        noisy_input_num_neighbors=5,
        encoder_output_num_neighbors=0,
        norm_type="offline_laplace_norm",
        weight_init=False,
        sr=16000,
        n_ftt=512,
        win_length=512,
        hop_length=256,
    ):
        """Fast FullSubNet.

        Notes:
            Here, the encoder, bottleneck, and decoder are corresponding to the F_l2m, S, and F_m2l models in the paper, respectively.
        """
        super().__init__(look_ahead, shrink_size, sequence_model, num_mels, encoder_input_size,
                            bottleneck_hidden_size, bottleneck_num_layers, noisy_input_num_neighbors,
                            encoder_output_num_neighbors, norm_type, weight_init, sr, n_ftt, win_length, hop_length)
        
        self.speaker_embedder = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Sigmoid()
        )
    def encoder_forward(self, audio):
        audio = audio.squeeze(1)
        mix_mag, _, noisy_real, noisy_imag = stft(audio, **self.stft_args)
        mix_mag = mix_mag.unsqueeze(1)
        assert mix_mag.dim() == 4
        mix_mag = F.pad(mix_mag, [0, self.look_ahead])  # Pad the look ahead
        batch_size, num_channels, num_freqs, num_frames = mix_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes a magnitude feature as the input."

        # Mel filtering
        mix_mel_mag = self.mel_scale(mix_mag)  # [B, C, F_mel, T]
        _, _, num_freqs_mel, _ = mix_mel_mag.shape

        # F_l2m
        enc_input = self.norm(mix_mel_mag).reshape(batch_size, -1, num_frames)
        enc_output = self.encoder(enc_input).reshape(batch_size, num_channels, -1, num_frames)  # [B, C, F, T]

        # Unfold - noisy spectrogram, [B, N=F, C, F_s, T]
        mix_mel_unfold_mag = self.freq_unfold(mix_mel_mag, num_neighbors=self.noisy_input_num_neighbors)  # [B, F_mel, C, F_sub, T]
        mix_mel_unfold_mag = mix_mel_unfold_mag.reshape(batch_size, self.num_mels, self.noisy_input_num_neighbors * 2 + 1, num_frames)  # [B, F_mel, F_sub, T]

        # Unfold - full-band model's output, [B, N=F, C, F_f, T], where N is the number of sub-band units
        enc_output_unfold_mel = self.freq_unfold(enc_output, num_neighbors=self.enc_output_num_neighbors)  # [B, F_mel, C, F_sub, T]
        enc_output_unfold_mel = enc_output_unfold_mel.reshape(batch_size, self.num_mels, self.enc_output_num_neighbors * 2 + 1, num_frames)  # [B, F_mel, F_sub, T]

        # Bottleneck (S)
        bn_input = torch.cat([mix_mel_unfold_mag, enc_output_unfold_mel], dim=2)
        num_sb_unit_freqs = bn_input.shape[2]

        # Bottleneck - time downsampling
        bn_input_shrink = self.real_time_downsampling(bn_input)  # [B, F_mel, F_sub_1 + F_sub_2, T // shrink_size]
        bn_input_shrink = self.norm(bn_input_shrink)  # [B, F_mel, F_sub_1 + F_sub_2, T // shrink_size]
        bn_input_shrink = bn_input_shrink.reshape(batch_size * self.num_mels, num_sb_unit_freqs, -1)  # [B * F_mel, F_sub_1 + F_sub_2, T // shrink_size]
        bn_output_shrink = self.bottleneck(bn_input_shrink)  # [B * F_mel, 1, T // shrink_size]
        bn_output_shrink = bn_output_shrink.reshape(batch_size, self.num_mels, 1, -1).permute(0, 2, 1, 3)  # [B, 1, F_mel, T // shrink_size]
        bn_output = self.real_time_upsampling(bn_output_shrink, target_len=num_frames)  # [B, 1, F_mel, T]
        output = torch.cat([enc_output, bn_output], dim=2)
        return output, batch_size, num_freqs, num_frames, noisy_real, noisy_imag    
    

    def subbands_reweighting(self, noisy, reference):
        b, c, f, t = reference.shape
        reference = reference.reshape(b*c, f, t)
        print(reference.shape)
        reference = F.adaptive_avg_pool1d(reference, 1).squeeze(-1)
        print(reference.shape)
        speaker_embedding = self.speaker_embedder(reference).unsqueeze(-1).reshape(b, c, f, 1)
        return noisy * speaker_embedding

    # fmt: off
    def forward(self, data):
        """Forward pass.

        Args:
            data: dictionary containing the noisy waveform.

        Returns:
            The real part and imag part of the enhanced spectrogram with shape [B, 2, F, T].

        Notes:
            noisy: noisy waveform
            mix_mag: noisy magnitude spectrogram with shape [B, 1, F, T].
            B - batch size
            C - channel
            F - frequency
            F_mel - mel frequency
            T - time
            F_s - sub-band frequency
        """
        noisy = data['noisy']
        reference = data['reference']
        noisy_subbands, batch_size, num_freqs, num_frames, noisy_real, noisy_imag = self.encoder_forward(noisy)
        reference_subbands = self.encoder_forward(reference)[0]

        dec_input = self.subbands_reweighting(noisy_subbands, reference_subbands)
        # F_ml2
        dec_input = dec_input.reshape(batch_size, -1, num_frames)
        decoder_lstm_output = self.decoder_lstm(dec_input)  # [B * C, F * 2, T]
        dec_output = decoder_lstm_output.reshape(batch_size, 2, num_freqs, num_frames)

        # Output
        output = dec_output[:, :, :, self.look_ahead:]
        output = output.permute(0, 2, 3, 1)
        
        # Full band CRM mask
        enhanced = self.full_band_crm_mask(output, noisy, noisy_real, noisy_imag)
        enhanced = enhanced.unsqueeze(1)
        return enhanced, output
        
# fmt: on
if __name__ == "__main__":
    import time
    import argparse
    from torchinfo import summary
    
    args = argparse.ArgumentParser()
    args.add_argument("--source", type=str, help='Source wav file path')
    args.add_argument("--target", type=str, help='Target wav file path')
    args.add_argument("--checkpoint", type=str, help='Model checkpoint path')
    args = args.parse_args()

    with torch.no_grad():
        if args.source:
            noisy, sr = audio.load(args.source)
            noisy = noisy.unsqueeze(0)
        else:
            noisy = torch.rand(1, 1, 3*160000)
            reference = torch.rand(1, 1, 3 * 160000)
        model = PersonalizedFastFullSubnet()
        # Load the updated state dict into the new model
        if args.checkpoint:
            old_state_dict = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(old_state_dict)
        
        start = time.time()
        enhanced, cRM = model({'noisy': noisy, 'reference': reference})
        print(f'input shape: {noisy.shape}, output shape: {enhanced.shape}')
        end = time.time()
        print(f'inference time: {end - start:.4f} s')
        if args.target:
            for i in range(enhanced.size(0)):
                audio.save(args.target + f'_{i}.wav', enhanced[i], sr)
        summary(model, input_data={'data':{'noisy': noisy, 'reference': reference}}, device="cpu")
