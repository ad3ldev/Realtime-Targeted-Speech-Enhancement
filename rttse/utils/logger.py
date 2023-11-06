"""
Code from BasicSR
"""

import datetime
import logging
import time

from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

initialized_logger = {}


class AvgTimer():

    def __init__(self, window=200):
        self.window = window  # average window
        self.current_time = 0
        self.total_time = 0
        self.count = 0
        self.avg_time = 0
        self.start()

    def start(self):
        self.start_time = self.tic = time.time()

    def record(self):
        self.count += 1
        self.toc = time.time()
        self.current_time = self.toc - self.tic
        self.total_time += self.current_time
        # calculate average time
        self.avg_time = self.total_time / self.count

        # reset
        if self.count > self.window:
            self.count = 0
            self.total_time = 0

        self.tic = time.time()

    def get_current_time(self):
        return self.current_time

    def get_avg_time(self):
        return self.avg_time


# class MessageLogger():
#     """Message logger for printing.

#     Args:
#         opt (dict): Config. It contains the following keys:
#             name (str): Exp name.
#             logger (dict): Contains 'print_freq' (str) for logger interval.
#             train (dict): Contains 'total_iter' (int) for total iters.
#             use_tb_logger (bool): Use tensorboard logger.
#         start_iter (int): Start iter. Default: 1.
#         tb_logger (obj:`tb_logger`): Tensorboard logger. Default： None.
#     """

#     def __init__(self, opt, start_iter=1, tb_logger=None):
#         self.exp_name = opt['name']
#         self.interval = opt['logger']['print_freq']
#         self.start_iter = start_iter
#         self.max_iters = opt['train']['total_iter']
#         self.use_tb_logger = opt['logger']['use_tb_logger']
#         self.tb_logger = tb_logger
#         self.start_time = time.time()
#         self.logger = get_root_logger()

#     def reset_start_time(self):
#         self.start_time = time.time()


#     def __call__(self, log_vars):
#         """Format logging message.

#         Args:
#             log_vars (dict): It contains the following keys:
#                 epoch (int): Epoch number.
#                 iter (int): Current iter.
#                 lrs (list): List for learning rates.

#                 time (float): Iter time.
#                 data_time (float): Data time for each iter.
#         """
#         # epoch, iter, learning rates
#         epoch = log_vars.pop('epoch')
#         current_iter = log_vars.pop('iter')
#         lrs = log_vars.pop('lrs')

#         message = (f'[{self.exp_name[:5]}..][epoch:{epoch:3d}, iter:{current_iter:8,d}, lr:(')
#         for v in lrs:
#             message += f'{v:.3e},'
#         message += ')] '

#         # time and estimated time
#         if 'time' in log_vars.keys():
#             iter_time = log_vars.pop('time')
#             data_time = log_vars.pop('data_time')

#             total_time = time.time() - self.start_time
#             time_sec_avg = total_time / (current_iter - self.start_iter + 1)
#             eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
#             eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
#             message += f'[eta: {eta_str}, '
#             message += f'time (data): {iter_time:.3f} ({data_time:.3f})] '

#         # other items, especially losses
#         for k, v in log_vars.items():
#             message += f'{k}: {v:.4e} '
#             # tensorboard logger
#             if self.use_tb_logger and 'debug' not in self.exp_name:
#                 if k.startswith('l_'):
#                     self.tb_logger.add_scalar(f'losses/{k}', v, current_iter)
#                 else:
#                     self.tb_logger.add_scalar(k, v, current_iter)
#         self.logger.info(message)

#     @property
#     def name(self):
#         return self.exp_name

#     @property
#     def version(self):
#         # Return the experiment version, int or str.
#         return "0.1"

#     @rank_zero_only
#     def log_hyperparams(self, params):
#         # params is an argparse.Namespace
#         # your code to record hyperparameters goes here
#         self.logger.info(params)

#     @rank_zero_only
#     def log_metrics(self, metrics, step):
#         # metrics is a dictionary of metric names and values
#         # your code to record metrics goes here
#         pass

#     @rank_zero_only
#     def save(self):
#         # Optional. Any code necessary to save logger data goes here
#         pass

#     @rank_zero_only
#     def finalize(self, status):
#         # Optional. Any code that needs to be run after training
#         # finishes goes here
#         pass




def init_tb_logger(save_dir):
    # from torch.utils.tensorboard import SummaryWriter
    from pytorch_lightning.loggers import TensorBoardLogger
    tb_logger = TensorBoardLogger(save_dir, name='')
    return tb_logger


def init_wandb_logger(opt):
    """We now only use wandb to sync tensorboard log."""
    import wandb
    logger = get_root_logger()

    project = opt['logger']['wandb']['project']
    resume_id = opt['logger']['wandb'].get('resume_id')
    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.warning(f'Resume wandb logger with id={wandb_id}.')
    else:
        wandb_id = wandb.util.generate_id()
        resume = 'never'

    wandb.init(id=wandb_id, resume=resume, name=opt['name'], config=opt, project=project, sync_tensorboard=True)

    logger.info(f'Use wandb logger with id={wandb_id}; project={project}.')


def get_root_logger(logger_name='rttse'):
    
    return logging.getLogger(logger_name)



# def get_env_info():
#     """Get environment information.

#     Currently, only log the software version.
#     """
#     import torch
#     import torchvision

#     from basicsr.version import __version__
#     msg = r"""
#                 ____                _       _____  ____
#                / __ ) ____ _ _____ (_)_____/ ___/ / __ \
#               / __  |/ __ `// ___// // ___/\__ \ / /_/ /
#              / /_/ // /_/ /(__  )/ // /__ ___/ // _, _/
#             /_____/ \__,_//____//_/ \___//____//_/ |_|
#      ______                   __   __                 __      __
#     / ____/____   ____   ____/ /  / /   __  __ _____ / /__   / /
#    / / __ / __ \ / __ \ / __  /  / /   / / / // ___// //_/  / /
#   / /_/ // /_/ // /_/ // /_/ /  / /___/ /_/ // /__ / /<    /_/
#   \____/ \____/ \____/ \____/  /_____/\____/ \___//_/|_|  (_)
#     """
#     msg += ('\nVersion Information: '
#             f'\n\tBasicSR: {__version__}'
#             f'\n\tPyTorch: {torch.__version__}'
#             f'\n\tTorchVision: {torchvision.__version__}')
#     return msg




