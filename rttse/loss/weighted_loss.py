from torch import nn

class WeightedLoss(nn.Module):
    def __init__(self, loss, weight=1):
        super(WeightedLoss, self).__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, input, target):
        loss = self.loss(input, target)
        if isinstance(loss, tuple):
            return tuple(map(lambda x: x * self.weight, loss))
        return loss * self.weight