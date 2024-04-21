from torch import nn

class WeightedLoss(nn.Module):
    def __init__(self, loss, weight=1):
        super(WeightedLoss, self).__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, input, target):
        loss = self.loss(input, target)
        print("Loss: ", loss)
        print("Loss shape: ", loss.shape)
        print("Weight: ", self.weight)
        return loss * self.weight