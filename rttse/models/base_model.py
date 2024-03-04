from collections import OrderedDict

import pytorch_lightning as pl
import hydra
from utils.console_logger import ConsoleLogger
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
        self.log_dict(loss_dict, on_step=True, on_epoch=True)
        return loss_dict['train/l_total']

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.net(x)

        metrics_dict = self.calculate_metrics(y_hat, y, 'val')

        self.log_dict(metrics_dict)


    def test_step(self,  batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)

        metrics_dict = self.calculate_metrics(y_hat, y, 'test')


        self.log_dict(metrics_dict)


    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.train.optim, params=self.get_bare_model().parameters())
        scheduler = hydra.utils.instantiate(self.hparams.train.scheduler, optimizer=optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    


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
