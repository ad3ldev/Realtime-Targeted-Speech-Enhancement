import hydra
import torch
import torchaudio

from collections import OrderedDict
from torch import nn
from models.base_model import BaseModel
from utils.fastfullsubnetutils import stft, build_complex_ideal_ratio_mask

class FullSubNetModel(BaseModel):
    def __init__(
            self, 
            net, 
            initial_weights=None, 
            strict=True,
            n_ftt=512, 
            win_length=512, 
            hop_length=256
        ) -> None:
        super().__init__(net)
        if initial_weights is not None:
            initial_weights = torch.load(initial_weights)
            self.net.load_state_dict(initial_weights, strict=strict) 
        self.stft_args = {
            'n_fft': n_ftt,
            'win_length': win_length,
            'hop_length': hop_length
        }
    
    def setup_training(self, cfg):
        self.cfg = cfg
        self.save_hyperparameters(cfg, logger=False)

        self.loss_weights = cfg['train'].get('loss_weights', {})    
        self.losses = nn.ModuleDict(hydra.utils.instantiate(cfg['train']['losses']))
        self.cIRM_losses = nn.ModuleDict(hydra.utils.instantiate(cfg['train']['cIRM_losses']))

        self.metrics = nn.ModuleDict(hydra.utils.instantiate(cfg['val']['metrics']))      
        self.cIRM_metrics = nn.ModuleDict(hydra.utils.instantiate(cfg['val']['cIRM_metrics']))  
        
        self.use_cIRM_losses = len(self.cIRM_losses) > 0
        self.use_cIRM_metrics = len(self.cIRM_metrics) > 0
    
    def setup_testing(self, cfg):
        self.cfg = cfg
        self.save_hyperparameters(cfg, logger=False)
        
        self.metrics = nn.ModuleDict(hydra.utils.instantiate(cfg['metrics']))      
        self.cIRM_metrics = nn.ModuleDict(hydra.utils.instantiate(cfg['cIRM_metrics']))  
        
        self.use_cIRM_metrics = len(self.cIRM_metrics) > 0
    

    def batch_adapter(self, batch):
        # print("batch:", batch)
        return batch[1:], batch[0]
    

    def training_step(self, batch, batch_idx):
        x, y = self.batch_adapter(batch)
        # print("x:", x, "y:", y)
        
        y_hat, cRM = self.net(x)
        
        loss_dict = self.calculate_loss(y_hat, y, 'train')
        
        if self.use_cIRM_losses:
            cIRM = self.calculate_cIRM(x[0], y)
            loss_dict = self.calculate_cIRM_loss(cIRM, cRM, 'train', loss_dict)
        
        self.log_dict(loss_dict, on_step=True, on_epoch=True, sync_dist=True)
        return loss_dict['train/l_total']
    
    def validation_step(self, batch, batch_idx):
        x, y = self.batch_adapter(batch)
        # print("x:", x, "y:", y)

        y_hat, cRM = self.net(x)

        metrics_dict = self.calculate_metrics(y_hat, y, 'val')
        if self.use_cIRM_metrics:
            cIRM = self.calculate_cIRM(x[0], y)
            metrics_dict = self.calculate_cIRM_metrics(cIRM, cRM, 'val', metrics_dict)

        self.log_dict(metrics_dict, sync_dist=True)
    
    def test_step(self,  batch, batch_idx):
        x, y = self.batch_adapter(batch)
        y_hat, cRM = self.net(x)

        metrics_dict = self.calculate_metrics(y_hat, y, 'test')
        if self.use_cIRM_metrics:
            cIRM = self.calculate_cIRM(x[1], y)
            metrics_dict = self.calculate_cIRM_metrics(cIRM, cRM, 'test', metrics_dict)
        
        if self.cfg.save_results:
            torchaudio.save(f"{self.cfg.save_dir}/{batch_idx}.wav", y_hat.squeeze(1).cpu(), self.cfg.sample_rate)

        self.log_dict(metrics_dict, sync_dist=True)

    
    def calculate_cIRM(self, noisy, clean) -> torch.Tensor:
        _, _, noisy_real, noisy_imag = stft(noisy, **self.stft_args)
        _, _, clean_real, clean_imag = stft(clean, **self.stft_args)
        cIRM = build_complex_ideal_ratio_mask(
            noisy_real, noisy_imag, clean_real, clean_imag
        )
        return cIRM
    
    def calculate_cIRM_loss(self, cIRM, cRM, phase, loss_dict = OrderedDict()):
        l_total = loss_dict[f'{phase}/l_total'] if f'{phase}/l_total' in loss_dict else 0
        for loss_name, loss_fn in self.cIRM_losses.items():
            loss_dict[f'{phase}/{loss_name}'] = loss_fn(cIRM, cRM) * self.loss_weights.get(loss_name, 1)
            # Check if the result is a tuple
            if isinstance(loss_dict[f'{phase}/{loss_name}'], tuple):
                loss_dict[f'{phase}/{loss_name}'] = sum(loss_dict[f'{phase}/{loss_name}'])
            l_total += loss_dict[f'{phase}/{loss_name}']

        loss_dict[f'{phase}/l_total'] = l_total
        return loss_dict
    
    def calculate_cIRM_metrics(self, cIRM, cRM, phase, metrics_dict = OrderedDict()):
        for metric_name, metric_fn in self.cIRM_metrics.items():
            result = metric_fn(cIRM, cRM)
            if isinstance(result, dict):
                metrics_dict.update({f'{phase}/{metric_name}/{k}': v for k, v in result.items()})
            else:
                metrics_dict[f'{phase}/{metric_name}'] = result
        return metrics_dict
    