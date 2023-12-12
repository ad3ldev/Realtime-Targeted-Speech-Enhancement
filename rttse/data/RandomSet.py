from torch import randn
from torch.utils.data import Dataset


class RandomDataSet(Dataset):
    def __init__(self, num_samples, transforms):
        super().__init__()
        self.data = randn(size=(num_samples, 3, 64, 64))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.data[index]
    




