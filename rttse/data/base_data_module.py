import pytorch_lightning as pl
from torch.utils.data import DataLoader
from hydra.utils import instantiate

# from RandomSet import RandomDataSet

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        print(opt)

    def setup(self, stage=None):
        self.train_dataset = self.opt['train']['dataset']
        self.val_dataset = self.opt['val']['dataset']


    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.opt['train']['dataloader'])
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.opt['val']['dataloader'])