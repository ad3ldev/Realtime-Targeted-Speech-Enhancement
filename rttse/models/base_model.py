from collections import OrderedDict
import pytorch_lightning as pl
import hydra

class BaseModel(pl.LightningModule):
    # def __init__(self) -> None:
    #     super().__init__()


    def __init__(self, net) -> None:
        super().__init__()
        self.net = net

    def network_to_string(self):
        return str(self.net)

    def setup_training(self, cfg):
        print(cfg)
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        # self.net = hydra.utils.instantiate(cfg['net'])
        # self.losses = hydra.utils.instantiate(cfg['train']['losses'])
        # self.metrics = hydra.utils.instantiate(cfg['val']['metrics'])

    def calculate_loss(self, y_hat, y, phase):
        loss_dict = OrderedDict()
        l_total = 0
        for loss_name, loss_fn in self.losses.items():
            loss_dict[f'{phase}/{loss_name}'] = loss_fn(y_hat, y)
            l_total += loss_dict[loss_name]

        loss_dict['l_total'] = l_total
        return loss_dict

    def calculate_metrics(self, y_hat, y, phase):
        metrics_dict = OrderedDict()
        for metric_name, metric_fn in self.metrics.items():
            metrics_dict[f'{phase}/{metric_name}'] = metric_fn(y_hat, y)

        return metrics_dict


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)

        loss_dict = self.calculate_loss(y_hat, y, 'train')
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True)
        return loss_dict['l_total']

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)

        loss_dict = self.calculate_loss(y_hat, y, 'val')

        metrics_dict = self.calculate_metrics(y_hat, y)

        logs_dict = {**loss_dict, **metrics_dict}

        self.log_dict(logs_dict)


    def test_step(self,  batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)

        loss_dict = self.calculate_loss(y_hat, y, 'test')

        metrics_dict = self.calculate_metrics(y_hat, y)

        logs_dict = {**loss_dict, **metrics_dict}

        self.log_dict(logs_dict)


    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.train.optim, params=self.parameters())
        scheduler = hydra.utils.instantiate(self.hparams.train.scheduler, optimizer=optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    # def on_train_start(self):
    #     # Proper logging of hyperparams and metrics in TB
    #     self.logger.log_hyperparams(self.hparams, {"loss/val": 0, "accuracy/val": 0, "accuracy/test": 0})