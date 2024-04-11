# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import math
import time

import torch as th
from torch import nn
from torch.nn import functional as F

from rttse.utils.demucs_utils import downsample2, upsample2, capture_init

class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class PDenoiser(nn.Module):
    """
    PDenoiser personalized speech enhancement model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.
        - sample_rate (float): sample_rate used for training the model.

    """
    @capture_init
    def __init__(self,
                 chin=1,
                 chout=1,
                 reference_embedding=192,
                 hidden=48,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 causal=True,
                 resample=4,
                 growth=2,
                 max_hidden=10_000,
                 normalize=True,
                 glu=True,
                 rescale=0.1,
                 floor=1e-3,
                 sample_rate=16_000):

        super().__init__()
        if resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize
        self.sample_rate = sample_rate

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.speaker_embedding = nn.Sequential(
            nn.Linear(reference_embedding, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, chin),
            nn.LayerNorm(chin),
            nn.ReLU())
        
        self.lstm = BLSTM(chin, bi=not causal)
        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, mix, reference):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        # print(x.shape)
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)


        reference = self.speaker_embedding(reference).unsqueeze(-1)
        # print(reference.shape)  
        # print(x.shape)

        x = x * reference
        x = x.permute(2, 0, 1)

        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        x = x[..., :length]
        return std * x


def fast_conv(conv, x):
    """
    Faster convolution evaluation if either kernel size is 1
    or length of sequence is 1.
    """
    batch, chin, length = x.shape
    chout, chin, kernel = conv.weight.shape
    assert batch == 1
    if kernel == 1:
        x = x.view(chin, length)
        out = th.addmm(conv.bias.view(-1, 1),
                       conv.weight.view(chout, chin), x)
    elif length == kernel:
        x = x.view(chin * kernel, 1)
        out = th.addmm(conv.bias.view(-1, 1),
                       conv.weight.view(chout, chin * kernel), x)
    else:
        out = conv(x)
    return out.view(batch, chout, -1)


# def test():
#     import argparse
#     parser = argparse.ArgumentParser(
#         "denoiser.demucs",
#         description="Benchmark the streaming Demucs implementation, "
#                     "as well as checking the delta with the offline implementation.")
#     parser.add_argument("--depth", default=5, type=int)
#     parser.add_argument("--resample", default=4, type=int)
#     parser.add_argument("--hidden", default=48, type=int)
#     parser.add_argument("--sample_rate", default=16000, type=float)
#     parser.add_argument("--device", default="cpu")
#     parser.add_argument("-t", "--num_threads", type=int)
#     parser.add_argument("-f", "--num_frames", type=int, default=1)
#     args = parser.parse_args()
#     if args.num_threads:
#         th.set_num_threads(args.num_threads)
#     sr = args.sample_rate
#     sr_ms = sr / 1000
#     demucs = PDenoiser(depth=args.depth, hidden=args.hidden, resample=args.resample).to(args.device)
#     x = th.randn(1, int(sr * 4)).to(args.device)
#     out = demucs(x[None])[0]
#     streamer = DemucsStreamer(demucs, num_frames=args.num_frames)
#     out_rt = []
#     frame_size = streamer.total_length
#     with th.no_grad():
#         while x.shape[1] > 0:
#             out_rt.append(streamer.feed(x[:, :frame_size]))
#             x = x[:, frame_size:]
#             frame_size = streamer.demucs.total_stride
#     out_rt.append(streamer.flush())
#     out_rt = th.cat(out_rt, 1)
#     model_size = sum(p.numel() for p in demucs.parameters()) * 4 / 2**20
#     initial_lag = streamer.total_length / sr_ms
#     tpf = 1000 * streamer.time_per_frame
#     print(f"model size: {model_size:.1f}MB, ", end='')
#     print(f"delta batch/streaming: {th.norm(out - out_rt) / th.norm(out):.2%}")
#     print(f"initial lag: {initial_lag:.1f}ms, ", end='')
#     print(f"stride: {streamer.stride * args.num_frames / sr_ms:.1f}ms")
#     print(f"time per frame: {tpf:.1f}ms, ", end='')
#     print(f"RTF: {((1000 * streamer.time_per_frame) / (streamer.stride / sr_ms)):.2f}")
#     print(f"Total lag with computation: {initial_lag + tpf:.1f}ms")


# if __name__ == "__main__":
#     # test()
#     # model = PDenoiser()
#     # model(th.randn(1, 1, 16000), th.randn(1, 192))
