"""
Code from BasicSR
"""

import logging

def init_tb_logger(save_dir):
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



def get_env_info():
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    import torchvision
    import pytorch_lightning as pl
    import hydra
    
    msg = ('\nVersion Information: '
            f'\n\tPyTorch: {torch.__version__}'
            f'\n\tTorchVision: {torchvision.__version__}'
            f'\n\tPyTorch-Lightning: {pl.__version__}'
            f'\n\tHydra: {hydra.__version__}')
    return msg




