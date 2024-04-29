import torch.nn as nn

class EmtpyNet(nn.Module):
    def __init__(self):
        super(EmtpyNet, self).__init__()

    def forward(self, x, **kwargs):
        return x