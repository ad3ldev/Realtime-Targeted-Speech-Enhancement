import hydra
from omegaconf import OmegaConf

from pytorch_lightning import Trainer, seed_everything
from utils.logger import get_env_info, init_logging
from utils.logger import get_root_logger



def setup_datasets(data_cfg):
    data_loader = hydra.utils.instantiate(data_cfg)
    return data_loader


def setup_model(model_cfg, testing_cfg):
    model = hydra.utils.instantiate(model_cfg)
    model.setup_training(testing_cfg)

    model.print_netowrk()

    return model

def setup_trainer(trainer_cfg):
    seed_everything(trainer_cfg.manual_seed)
    trainer = Trainer(**trainer_cfg['trainer_args'])
    return trainer, trainer_cfg.get('checkpoint_path')



@hydra.main(version_base=None, config_path="../config", config_name="test_config")
def test_pipeline(cfg):
    logger = get_root_logger()

    logger.info(get_env_info())


    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    tb_logger = init_logging(cfg)

    trainer, ckpt_path = setup_trainer(cfg.trainer, tb_logger)

    data_loader = setup_datasets(cfg.data)

    model = setup_model(cfg.model, cfg.testing)


    trainer.test(model, data_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    test_pipeline()
