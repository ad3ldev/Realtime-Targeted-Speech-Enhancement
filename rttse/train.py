from time import sleep
import hydra
from omegaconf import OmegaConf

from pytorch_lightning import Trainer, seed_everything
from utils.logger import get_env_info, init_logging
from utils.logger import get_root_logger



def setup_datasets(data_cfg):
    data_loader = hydra.utils.instantiate(data_cfg)
    return data_loader


def setup_model(model_cfg, training_cfg):
    model = hydra.utils.instantiate(model_cfg)
    model.setup_training(training_cfg)

    model.print_netowrk()

    model.configure_optimizers()

    return model

def setup_trainer(trainer_cfg, tb_logger):
    seed_everything(trainer_cfg.manual_seed)
    callbacks = hydra.utils.instantiate(trainer_cfg.callbacks)
    callbacks = [list(cb.values())[0] for cb in callbacks]
    trainer = Trainer(**trainer_cfg['trainer_args'], logger=tb_logger, callbacks=callbacks)
    return trainer, trainer_cfg.get('checkpoint_path')



@hydra.main(version_base=None, config_path="../config", config_name="config")
def train_pipeline(cfg):
    logger = get_root_logger()

    logger.info(get_env_info())


    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    tb_logger = init_logging(cfg)

    trainer, ckpt_path = setup_trainer(cfg.trainer, tb_logger)

    data_loader = setup_datasets(cfg.data)

    model = setup_model(cfg.model, cfg.training)


    trainer.fit(model, data_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    # import logging
    # logging.disable(logging.CRITICAL)
    train_pipeline()
