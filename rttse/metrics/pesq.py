from torch import nn
from pesq import pesq

class PESQ(nn.Module):
    def __init__(self, sample_rate=16000, mode='wb'):
        super(PESQ, self).__init__()
        self.sample_rate = sample_rate
        self.mode = mode

    def forward(self, x, y):
        return pesq(self.sample_rate, x, y, self.mode)