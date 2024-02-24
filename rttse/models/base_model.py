from collections import OrderedDict
from datetime import datetime
from typing import Any
import pytorch_lightning as pl
import hydra
from utils.time import format_seconds
from utils.dist_utils import master_only
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from utils.logger import get_root_logger

class BaseModel(pl.LightningModule):
    def __init__(self, net) -> None:
        super().__init__()
        self.net = net

    def get_bare_model(self):
        return self.net

    def network_to_string(self):
        return str(self.get_bare_model())

    def setup_training(self, cfg):
        self.cfg = cfg
        self.save_hyperparameters(cfg, logger=False)

        self.loss_weights = cfg['train'].get('loss_weights', {})    
        self.losses = hydra.utils.instantiate(cfg['train']['losses'])

        self.metrics = hydra.utils.instantiate(cfg['val']['metrics'])

    @rank_zero_only
    def print_netowrk(self, stage=None):
        logger = get_root_logger()

        logger.info(f"\n{self.network_to_string()}\n")

    

    def calculate_loss(self, y_hat, y, phase):
        loss_dict = OrderedDict()
        l_total = 0
        for loss_name, loss_fn in self.losses.items():
            # loss_name, loss_fn = list(loss.items())[0]
            loss_dict[f'{phase}/{loss_name}'] = loss_fn(y_hat, y) * self.loss_weights.get(loss_name, 1)
            l_total += loss_dict[f'{phase}/{loss_name}']

        if len(self.losses) > 1: loss_dict[f'{phase}/l_total'] = l_total
        return loss_dict

    def calculate_metrics(self, y_hat, y, phase):
        metrics_dict = OrderedDict()
        for metric_name, metric_fn in self.metrics.items():
            # metric_name, metric_fn = list(metric.items())[0]
            metrics_dict[f'{phase}/{metric_name}'] = metric_fn(y_hat, y)

        return metrics_dict
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self.net(x)

        loss_dict = self.calculate_loss(y_hat, y, 'train')
        self.log_dict(loss_dict, on_step=True, on_epoch=True, logger=True)
        return loss_dict['train/l_total']

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.net(x)

        metrics_dict = self.calculate_metrics(y_hat, y, 'val')

        self.log_dict(metrics_dict, logger=True)


    def test_step(self,  batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)

        metrics_dict = self.calculate_metrics(y_hat, y, 'test')


        self.log_dict(metrics_dict, logger=True)


    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.train.optim, params=self.get_bare_model().parameters())
        scheduler = hydra.utils.instantiate(self.hparams.train.scheduler, optimizer=optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    


    def on_train_start(self) -> None:
        get_root_logger().info(f'Training Started...')
        self.train_tic = datetime.now()


    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            eta = self.estimate_remaining_time()
            losses_str = self.get_loss_str()
            lrs = self.get_lrs_str()
            get_root_logger().info(f'[epoch: {self.trainer.current_epoch}, iter: {self.trainer.global_step}, lr: {lrs}] [eta: {eta}] {losses_str}')

    def get_lrs_str(self):
        optims = self.optimizers()
        if optims is list:
            lrs = (f"{opt.param_groups[0]['lr']:.4f}" for opt in optims)
        else:
            lrs = f"{optims.param_groups[0]['lr']:.4f}"
        return lrs

    def get_loss_str(self):
        metrics = self.trainer.callback_metrics
        loss_dict = {k: v for k, v in metrics.items() if 'train' in k and 'step' in k}
        losses_str = ""
        for k, v in loss_dict.items():
            losses_str += f"{k}: {v:.4f}, "

        losses_str = losses_str[:-2]
        return losses_str

    def estimate_remaining_time(self):
        duration = (datetime.now() - self.train_tic).seconds / self.trainer.log_every_n_steps
        eta = format_seconds(duration * (self.trainer.max_steps - self.trainer.global_step))
        self.train_tic = datetime.now()
        return eta

    def on_validation_epoch_start(self):
        get_root_logger().info(f'Validation Started...')
        self.val_tic = datetime.now()


    def on_validation_epoch_end(self):
        duration = format_seconds((datetime.now() - self.val_tic).seconds)
        metrics_str = self.get_metrics_str()
        get_root_logger().info(f'Validation Results: {metrics_str} and took {duration}')
        self.train_tic = datetime.now()

    def get_metrics_str(self):
        metrics = self.trainer.callback_metrics
        metrics = {k: v for k, v in metrics.items() if 'val' in k}
        metrics_str = ""
        for k, v in metrics.items():
            metrics_str += f"{k}: {v:.4f}, "
        
        metrics_str = metrics_str[:-2]
        return metrics_str
