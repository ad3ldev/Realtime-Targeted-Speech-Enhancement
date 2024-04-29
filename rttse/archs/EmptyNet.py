import torch.nn as nn

class EmptyNet(nn.Module):
    def __init__(self):
        super(EmptyNet, self).__init__()

    def forward(self, noisy_audio, labels):
        return noisy_audio