from time import sleep
import hydra
from omegaconf import OmegaConf

from pytorch_lightning import Trainer, seed_everything
from utils.logger import init_wandb_logger, init_tb_logger
from utils.logger import get_root_logger



def init_tb_loggers(cfg):

    opt = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) \
        and (opt['logger']['wandb'].get('project') is not None) \
        and ('debug' not in opt['name']):

        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')

        init_wandb_logger(opt)

    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(save_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    return tb_logger



def setup_datasets(cfg):
    data_loader = hydra.utils.instantiate(cfg.data)
    return data_loader


def setup_model(cfg):
    model = hydra.utils.instantiate(cfg.model)
    model.setup_training(cfg.training)

    model.configure_optimizers()

    return model

def setup_trainer(cfg, tb_logger):
    callbacks = hydra.utils.instantiate(cfg.callbacks)
    callbacks = [list(cb.values())[0] for cb in callbacks]
    trainer = Trainer(**cfg['trainer'], logger=tb_logger, callbacks=callbacks)
    return trainer



@hydra.main(version_base=None, config_path="../config", config_name="config")
def train_pipeline(cfg):

    seed_everything(cfg.manual_seed)

    tb_logger = init_tb_loggers(cfg)
    logger = get_root_logger()

    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    data_loader = setup_datasets(cfg)

    model = setup_model(cfg)

    trainer = setup_trainer(cfg, tb_logger)

    trainer.fit(model, data_loader, ckpt_path=cfg.get('checkpoint_path'))


if __name__ == "__main__":
    train_pipeline()
