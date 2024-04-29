from collections import OrderedDict

import pytorch_lightning as pl
import hydra
from utils.console_logger import ConsoleLogger
from torch import nn
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from utils.logger import get_root_logger

class BaseModel(pl.LightningModule):
    def __init__(self, net) -> None:
        super().__init__()
        self.net = net
        self.console_logger = ConsoleLogger()

    def get_bare_model(self):
        return self.net

    def network_to_string(self):
        return str(self.get_bare_model())

    def setup_training(self, cfg):
        self.cfg = cfg
        self.save_hyperparameters(cfg, logger=False)
   
        self.losses = nn.ModuleDict(hydra.utils.instantiate(cfg['train']['losses']))

        self.metrics = nn.ModuleDict(hydra.utils.instantiate(cfg['val']['metrics']))

    @rank_zero_only
    def print_netowrk(self, stage=None):
        logger = get_root_logger()

        logger.info(f"\n{self.network_to_string()}\n")

    

    def calculate_loss(self, y_hat, y, phase):
        loss_dict = OrderedDict()
        l_total = 0
        for loss_name, loss_fn in self.losses.items():
            # loss_name, loss_fn = list(loss.items())[0]
            loss_dict[f'{phase}/{loss_name}'] = loss_fn(y_hat, y)
            # Check if the result is a tuple
            if isinstance(loss_dict[f'{phase}/{loss_name}'], tuple):
                loss_dict[f'{phase}/{loss_name}'] = sum(loss_dict[f'{phase}/{loss_name}'])
            elif isinstance(loss_dict[f'{phase}/{loss_name}'], dict):
                loss_dict.update({f'{phase}/{loss_name}/{k}': v for k, v in loss_dict[f'{phase}/{loss_name}'].items()})
            l_total += loss_dict[f'{phase}/{loss_name}']

        loss_dict[f'{phase}/l_total'] = l_total
        return loss_dict

    def calculate_metrics(self, y_hat, y, phase):
        metrics_dict = OrderedDict()
        for metric_name, metric_fn in self.metrics.items():
            result = metric_fn(y_hat, y)
            if isinstance(result, tuple):
                metrics_dict[f'{phase}/{metric_name}'] = sum(result)
            elif isinstance(result, dict):
                metrics_dict.update({f'{phase}/{metric_name}/{k}': v for k, v in result.items()})
            else:
                metrics_dict[f'{phase}/{metric_name}'] = result
        return metrics_dict

    def batch_adapter(self, batch):
        return batch, batch['clean']
    

    def training_step(self, batch, batch_idx):
        x, y = self.batch_adapter(batch)
        
        y_hat = self.net(x)

        loss_dict = self.calculate_loss(y_hat, y, 'train')
        self.log_dict(loss_dict, on_step=True, on_epoch=True, sync_dist=True)
        return loss_dict['train/l_total']

    def validation_step(self, batch, batch_idx):
        x, y = self.batch_adapter(batch)

        y_hat = self.net(x)

        metrics_dict = self.calculate_metrics(y_hat, y, 'val')

        self.log_dict(metrics_dict, sync_dist=True)

    def test_step(self,  batch, batch_idx):
        x, y = self.batch_adapter(batch)
        y_hat = self.net(x)

        metrics_dict = self.calculate_metrics(y_hat, y, 'test')


        self.log_dict(metrics_dict, sync_dist=True)


    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.train.optim, params=self.get_bare_model().parameters())
        if 'scheduler' in self.hparams.train:
            scheduler = hydra.utils.instantiate(self.hparams.train.scheduler, optimizer=optimizer)
            if 'monitor' in self.hparams.train:
                monitor = self.hparams.train.monitor
                return [optimizer], [{"scheduler": scheduler, "monitor": monitor, "interval": "epoch"}]
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return [optimizer]

    def on_train_start(self) -> None:
        get_root_logger().info(f'Training Started...')
        self.console_logger.train_tic()


    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            self.console_logger.log_train_step(self.trainer, self.trainer.callback_metrics, self.optimizers())

    
    def on_validation_epoch_start(self):
        get_root_logger().info(f'Validation Started...')
        self.console_logger.val_tic()


    def on_validation_epoch_end(self):
        self.console_logger.log_validation_result(self.trainer, self.trainer.callback_metrics)
        self.console_logger.train_tic()
