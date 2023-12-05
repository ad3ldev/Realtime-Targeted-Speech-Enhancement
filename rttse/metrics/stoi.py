from torch import nn
from stoi import stoi

class STOI(nn.Module):
    def __init__(self, sample_rate=16000):
        super(STOI, self).__init__()
        self.sample_rate = sample_rate

    def forward(self, x, y):
        return stoi(x, y, self.sample_rate)