import hydra
import torch
import os

from omegaconf import OmegaConf

from pytorch_lightning import Trainer, seed_everything
from utils.logger import get_env_info, init_logging
from utils.logger import get_root_logger

def setup_testing(testing_cfg):
    if testing_cfg.save_dir is not None:
        os.makedirs(testing_cfg.save_dir, exist_ok=True)

def setup_datasets(data_cfg):
    data_loader = hydra.utils.instantiate(data_cfg)
    return data_loader


def setup_model(model_cfg, testing_cfg):
    model = hydra.utils.instantiate(model_cfg)
    model.setup_testing(testing_cfg)

    model.print_netowrk()

    return model

def setup_checkpoint(testing_cfg):
    if testing_cfg.trainer_args['ckpt_path'] is None:
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chkpt = torch.load(testing_cfg.trainer_args['ckpt_path'], map_location=device)
    
    # Remove keys starting with 'losses.' from the state_dict and save it
    state_dict = {k: v for k, v in chkpt['state_dict'].items() if not k.startswith('losses.')}
    chkpt['state_dict'] = state_dict

    new_ckpt_path = testing_cfg.trainer_args['ckpt_path'] + '_without_losses'
    torch.save(chkpt, new_ckpt_path)
    testing_cfg.trainer_args['ckpt_path'] = new_ckpt_path
    
    return testing_cfg

@hydra.main(version_base=None, config_path="../config", config_name="test_config")
def test_pipeline(cfg):
    logger = get_root_logger()

    logger.info(get_env_info())


    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    seed_everything(cfg.testing.manual_seed)

    tb_logger = init_logging(cfg)

    trainer = Trainer(logger=tb_logger)

    data_loader = setup_datasets(cfg.data)

    model = setup_model(cfg.model, cfg.testing)
    
    cfg.testing = setup_checkpoint(cfg.testing)

    setup_testing(cfg.testing)
    
    trainer.test(model, data_loader, **cfg.testing.trainer_args)

if __name__ == "__main__":
    test_pipeline()
