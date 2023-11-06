from collections import OrderedDict
import pytorch_lightning as pl
import hydra

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
        self.save_hyperparameters(cfg)
        self.loss_weight = cfg['train']['losses_weights']    
        self.losses = hydra.utils.instantiate(cfg['train']['losses'])

        assert len(self.losses) == len(self.loss_weight), "size mismatch between losses and their weights"

        self.metrics = hydra.utils.instantiate(cfg['val']['metrics'])

    def setup(self, stage=None):
        logger = get_root_logger()

        logger.info(f"\n\t{self.network_to_string()}\n")

    

    def calculate_loss(self, y_hat, y, phase):
        loss_dict = OrderedDict()
        l_total = 0
        for idx, loss in enumerate(self.losses):
            loss_name, loss_fn = list(loss.items())[0]
            loss_dict[f'{phase}/{loss_name}'] = loss_fn(y_hat, y) * self.loss_weight[idx]
            l_total += loss_dict[f'{phase}/{loss_name}']

        if len(self.losses) > 1: loss_dict[f'{phase}/l_total'] = l_total
        return loss_dict

    def calculate_metrics(self, y_hat, y, phase):
        metrics_dict = OrderedDict()
        for metric in self.metrics:
            metric_name, metric_fn = list(metric.items())[0]
            metrics_dict[f'{phase}/{metric_name}'] = metric_fn(y_hat, y)

        return metrics_dict
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self.net(x)

        loss_dict = self.calculate_loss(y_hat, y, 'train')
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True)
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