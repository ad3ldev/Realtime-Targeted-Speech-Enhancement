from time import sleep
import hydra
from omegaconf import DictConfig, OmegaConf


from utils.logger import MessageLogger
from utils.logger import init_wandb_logger, init_tb_logger
from utils.logger import get_root_logger



def init_tb_loggers(cfg):

    opt = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get(
            'use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)

    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger()

    return tb_logger


def setup_datasets(cfg, logger):
    pass


def setup_model(cfg, logger):
    pass

def training_loop(cfg, model, dataset, logger, msg_logger):
    pass

@hydra.main(version_base=None, config_path="../config", config_name="config")
def train_pipeline(cfg):
    print(OmegaConf.to_yaml(cfg))

    tb_logger = init_tb_loggers(cfg)
    logger = get_root_logger()

    train_dataset, val_datasets = setup_datasets(cfg['dataset'], logger)
    model = setup_model(cfg['model'], logger)




    msg_logger = MessageLogger()


    training_loop(cfg, model, train_dataset, val_datasets, logger, msg_logger)

    for i in range(50):
        sleep(1)
        tb_logger.add_scalar("loss", i, i)
        logger.info(f'loss: {i}')


if __name__ == "__main__":
    train_pipeline()
