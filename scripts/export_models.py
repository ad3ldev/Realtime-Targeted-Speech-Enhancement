import os
import sys
import torch
import hydra
import pickle

rttse_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rttse')
sys.path.append(rttse_directory)

@hydra.main(version_base=None, config_path="../config", config_name="stream_config")
def main(cfg):
    output_dir = cfg.output_dir if cfg.output_dir else '.'
    if output_dir[-1] == '/' or output_dir[-1] == '\\':
        output_dir = output_dir[:-1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model: torch.nn.Module = hydra.utils.instantiate(cfg.model)
    
    # load weights from pytorch lightning checkpoint
    checkpoint = torch.load(cfg.checkpoint_path)
    # remove keys starting with 'losses.' and 'metrics.'
    checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if not k.startswith('losses.') and not k.startswith('metrics.')}
    model.load_state_dict(checkpoint['state_dict'])
    
    # save speaker_embedder
    speaker_embedder = model.net.speaker_embedder
    with open(f'{output_dir}/{cfg.speaker_embedder_name}.embedder', 'wb') as f:
        pickle.dump(speaker_embedder, f)
    print(f'Saved speaker embedder to {output_dir}/{cfg.speaker_embedder_name}.embedder')
    
    # save model
    model = model.net.speech_enhancer
    torch.save(model, f'{output_dir}/{cfg.speech_enhancer_name}.model')
    print(f'Saved speech enhancer to {output_dir}/{cfg.speech_enhancer_name}.model')
    
if __name__ == '__main__':
    main()