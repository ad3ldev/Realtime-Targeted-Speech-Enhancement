import torch.nn as nn
import torch.nn.functional as F
import torch

class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout=0.2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.dp = nn.Dropout(dropout)

    
    def forward(self, x):

        return self.act(self.bn(self.conv(self.dp(x))))


class Encoder(nn.Module):
    def __init__(self, in_channels, num_layers) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [Layer(in_channels * (1<<i), in_channels * (1 << (i+1)), 3, 1, 1) for i in range(num_layers)]
        )

    def forward(self, x):
        outputs = [x]
        for layer in self.layers:
            outputs.append(F.interpolate(layer(outputs[-1]), scale_factor=0.5))

        return outputs

        
class Decoder(nn.Module):
    def __init__(self, in_channels, num_layers) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [Layer(in_channels  // (1<<i), in_channels // (1 << (i+1)), 3, 1, 1) for i in range(num_layers)]
        )

    def forward(self, skips):
        i = 1
        for layer in self.layers:
            skips[-(i+1)] += F.interpolate(layer(skips[-i]), scale_factor=2)
            i += 1

        return skips[0]

class SimpleAE(nn.Module):
    def __init__(self, channels, embeding, num_layers) -> None:
        super().__init__()
        self.embeding = embeding
        self.expand = nn.Conv2d(channels, embeding, 1)
        self.encoder = Encoder(embeding, num_layers)
        self.decoder = Decoder(embeding * 2**num_layers, num_layers)
        self.shrink = nn.Conv2d(embeding, channels, 1)


    def forward(self, x):
        b, c, h, w = x.shape
        skips = self.encoder(self.expand(x))
        skips[0] = torch.zeros((b, self.embeding, h, w))
        return self.shrink(self.decoder(skips))
        

