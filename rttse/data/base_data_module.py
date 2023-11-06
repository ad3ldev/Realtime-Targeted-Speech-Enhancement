import pytorch_lightning as pl
from torch.utils.data import DataLoader

from utils.logger import get_root_logger


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

    def info():
        pass

    def setup(self, stage=None):
        self.train_dataset = self.opt['train']['dataset']
        self.val_dataset = self.opt['val']['dataset']
        logger = get_root_logger()
        logger.info("\nTraining Statistics:\n--------------------"
                f"\n\t# Number of Training Samples: {len(self.train_dataset)}"
                f"\n\t# Number of Validation Samples: {len(self.val_dataset)}"
                f"\n\t# Training Batch Size: {self.opt['train']['dataloader']['batch_size']}\n\n")


    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.opt['train']['dataloader'])
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.opt['val']['dataloader'])