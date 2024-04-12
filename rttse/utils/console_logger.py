from datetime import datetime
from utils.logger import get_root_logger
from utils.time import format_seconds
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class ConsoleLogger():
    @rank_zero_only
    def __init__(self):
        self.__root_logger = get_root_logger()
        self.train_tic()
        self.val_tic()


    def __get_lrs_str(self, optims):
        if optims is list:
            lrs = (f"{opt.param_groups[0]['lr']:.4f}" for opt in optims)
        else:
            lrs = f"{optims.param_groups[0]['lr']:.4f}"
        return lrs

    def __get_loss_str(self, metrics):
        loss_dict = {k: v for k, v in metrics.items() if 'train' in k and 'step' in k}
        losses_str = ""
        for k, v in loss_dict.items():
            losses_str += f"{k}: {v:.4f}, "

        losses_str = losses_str[:-2]
        return losses_str

    def __estimate_remaining_time(self, trainer):
        eta = "N/A"
        if trainer.max_steps > -1:
            duration = (datetime.now() - self.__train_tic_).seconds / trainer.log_every_n_steps
            eta = format_seconds(duration * (trainer.max_steps - trainer.global_step))
            self.train_tic()

        return eta

    def __get_metrics_str(self, metrics):
        # metrics = self.trainer.callback_metrics
        metrics = {k: v for k, v in metrics.items() if 'val' in k}
        metrics_str = ""
        for k, v in metrics.items():
            metrics_str += f"{k}: {v:.4f}, "
        
        metrics_str = metrics_str[:-2]
        return metrics_str
    
    @rank_zero_only
    def train_tic(self):
        self.__train_tic_ = datetime.now()

    @rank_zero_only
    def val_tic(self):
        self.__val_tic_ = datetime.now()

    @rank_zero_only
    def log_train_step(self, trainer, metrics, optims):
        eta = self.__estimate_remaining_time(trainer)
        losses_str = self.__get_loss_str(metrics)
        lrs = self.__get_lrs_str(optims)
        self.__root_logger.info(f'[epoch: {trainer.current_epoch}, iter: {trainer.global_step}, lr: {lrs}] [eta: {eta}] {losses_str}')

    @rank_zero_only
    def log_validation_result(self, trainer, metrics):
        duration = format_seconds((datetime.now() - self.__val_tic_).seconds)
        metrics_str = self.__get_metrics_str(metrics)
        self.__root_logger.info(f'[epoch: {trainer.current_epoch}] [val] [duration: {duration}] {metrics_str}')