import os
import numpy as np
import pandas as pd
from typing import Literal

import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

import torchaudio

from nemo.collections.asr.models import EncDecSpeakerLabelModel

speaker_embedder = EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large').eval()

for param in speaker_embedder.parameters():
    param.requires_grad = False

# speaker_embedder_fn = lambda x: 

class PDNSDataset(Dataset):
    """
    Create a Dataset for the PDNS dataset. The dataset is created by matching the speakers from the synthesized speakers csv file to the speakers csv files.
    """
    
    def __init__(
        self, 
        speaker_embedder,
        root = './', 
        synthesized_speakers_csv = './speakers.csv',
        reference_speakers_csv = './reference_speakers.csv',
        sr = 41000, 
        crop_length_sec = 0, 
        mode: Literal['all', 'ps', 'pn', 'psn'] = 'all',
        seed: int = 42,
        # reference_tensor: bool = False
        ):
        """ Creates a PDNSDataset object.

        Args:
            speaker_embedder: The speaker embedding function.
            root (str, optional): The root directory of PDNS that contains clean and noisy subdirectories. Defaults to './'.
            synthesized_speakers_csv (str, optional): The csv containing the synthesization sources of each generated audio clip. This is used to find the primary speaker of each clip. Defaults to './speakers.csv'.
            reference_speakers_csv (str, optional): The csv containing the speakers of all raw files from the dataset. This is used to get a clean audio clip related to a sepcific speaker. Defaults to './reference_speakers.csv'.
            sr (int, optional): The required sampling rate of all the data. Defaults to 41000.
            crop_length_sec (int, optional): The required length of each audio clip. If zero doesn't crop. Defaults to 0.
            mode (Literal[&#39;all&#39;, &#39;ps&#39;, &#39;pn&#39;, &#39;psn&#39;], optional): The noisy speech mode. Used for the selection of noisy audio clips. "all" uses all of the noisy clips, "ps" stands for primary and secondry, "pn" stands for primary and noisy, "psn" stands for primary secondary and noisy. Defaults to 'all'.
            seed (int, optional): The seed used to select the reference audio clip. Defaults to 42.
            reference_tensor (bool, optional): Whether the reference needed to be loaded as a tensor or not. Defaults to False.

        Raises:
            ValueError: In case of an unknown noisy file which doesn't start with 'primary', 'ps', or 'psn'.
        """
        super(PDNSDataset).__init__()
        
        torch.multiprocessing.set_start_method('spawn')

        self.rng = np.random.default_rng(seed) # May need to change this to torch seed
        
        self.crop_length_sec = crop_length_sec
        self.sr = sr
        # self.reference_tensor = reference_tensor
        self.speaker_embedder = speaker_embedder
        
        # Load reference speaker csv
        reference_speakers = pd.read_csv(reference_speakers_csv)
        reference_speakers = reference_speakers[reference_speakers['speaker_type'] == 'primary']
        self.reference_files = dict()
        for _, row in reference_speakers.iterrows():
            if row['speaker_id'] not in self.reference_files:
                self.reference_files[row['speaker_id']] = [row['filename']]
            else:
                self.reference_files[row['speaker_id']].append(row['filename'])
        
        # Load synthesized speaker csv
        synthesized_speakers_df = pd.read_csv(synthesized_speakers_csv)
        synthesized_primary_speakers = synthesized_speakers_df['primary_speaker'].tolist()

        clean_subdir = 'clean'
        noisy_subdir = 'noisy'
        
        clean_files = [os.path.join(root, clean_subdir, file) for file in os.listdir(os.path.join(root, clean_subdir))]
        noisy_files = [os.path.join(root, noisy_subdir, file) for file in os.listdir(os.path.join(root, noisy_subdir))]
        assert len(clean_files)*3 == len(noisy_files), "Number of clean and noisy files does not match" # 3 noisy files per clean file
        assert len(clean_files) == len(synthesized_primary_speakers), "Number of clean files and synthesized primary speakers does not match"
        
        noisy_ps_files = []
        noisy_pn_files = []
        noisy_psn_files = []
        for noisy_file in noisy_files:
            noisy_file = os.path.basename(noisy_file)
            if noisy_file.startswith('primary'):
                noisy_pn_files.append(noisy_file)
            elif noisy_file.startswith('ps'):
                noisy_ps_files.append(noisy_file)
            elif noisy_file.startswith('psn'):
                noisy_psn_files.append(noisy_file)
            else:
                raise ValueError(f"Unknown noise type for file {noisy_file}")
        
        if mode == 'ps':
            noisy_files = noisy_ps_files
        elif mode == 'pn':
            noisy_files = noisy_pn_files
        elif mode == 'psn':
            noisy_files = noisy_psn_files
        
        clean_files.sort()
        noisy_files.sort()
        
        if(len(clean_files)*3 == len(noisy_files)): # In case of 'all' mode, repeat the noisy files 3 times
            clean_files = clean_files*3
            synthesized_primary_speakers = synthesized_primary_speakers*3
        
        # Number of clean and noisy files should match at this point
        assert len(clean_files) == len(noisy_files), "Number of clean and noisy files does not match" 
        
        # Create a list of tuples of clean files, noisy files and primary speakers
        self.files = list(zip(clean_files, noisy_files, synthesized_primary_speakers))

    def __getitem__(self, n):
        file = self.files[n]
        clean_audio, clean_sr = torchaudio.load(file[0])
        noisy_audio, noisy_sr = torchaudio.load(file[1])
        
        # Select a random speaker from the clean speakers
        reference_file = self.rng.choice(self.reference_files[file[2]])
        speaker_embedding = speaker_embedder.get_embedding(reference_file)
        
        # Resample the audio to the desired sample rate
        if clean_sr != self.sr:
            clean_audio = torchaudio.transforms.Resample(orig_freq=clean_sr, new_freq=self.sr)(clean_audio)
        if noisy_sr != self.sr:
            noisy_audio = torchaudio.transforms.Resample(orig_freq=noisy_sr, new_freq=self.sr)(noisy_audio)
        
        clean_audio, noisy_audio = clean_audio.squeeze(0), noisy_audio.squeeze(0)
        assert len(clean_audio) == len(noisy_audio)

        crop_length = int(self.crop_length_sec * self.sr)
        assert crop_length < len(clean_audio)

        # Random crop
        if crop_length > 0:
            start = np.random.randint(low=0, high=len(clean_audio) - crop_length + 1)
            clean_audio = clean_audio[start:(start + crop_length)]
            noisy_audio = noisy_audio[start:(start + crop_length)]
        
        clean_audio, noisy_audio = clean_audio.unsqueeze(0), noisy_audio.unsqueeze(0)
        
        # data = {
        #     "clean": clean_audio,
        #     "noisy": noisy_audio,
        #     "reference_path": reference_file
        # }
        
        # Load reference audio if reference_tensor is True
        # if self.reference_tensor:
        #     reference_audio, reference_sr = torchaudio.load(reference_file)
        #     if reference_sr != self.sr:
        #         reference_audio = torchaudio.transforms.Resample(orig_freq=reference_sr, new_freq=self.sr)(reference_audio)
        #     data["reference"] = reference_audio
        
        return (noisy_audio, speaker_embedding.squeeze()), clean_audio

    def __len__(self):
        return len(self.files)
    
    # @staticmethod
    # def collate_fn(batch):
    #     """Collate function for the dataloader

    #     Args:
    #         batch : List of data returned by __getitem__

    #     Returns:
    #         dict: A dictionary containing the clean, noisy, reference audio tensors if exist and the reference path list.
    #     """
    #     return {
    #         "clean": torch.stack([data["clean"] for data in batch]),
    #         "noisy": torch.stack([data["noisy"] for data in batch]),
    #         "reference_path": [data["reference_path"] for data in batch],
    #         "reference": torch.stack([data["reference"] for data in batch]) if "reference" in batch[0] else None
    #     }


def load_PDNSDataset(root, synthesized_speakers_csv, reference_speakers_csv, crop_length_sec, batch_size, sample_rate, num_gpus=1):
    """
    Get dataloader with distributed sampling
    """
    dataset = PDNSDataset(speaker_embedder=None, root=root, crop_length_sec=crop_length_sec, synthesized_speakers_csv=synthesized_speakers_csv, reference_speakers_csv=reference_speakers_csv, sr=sample_rate)                                                       
    kwargs = {"batch_size": batch_size, "num_workers": 4, "pin_memory": False, "drop_last": False}

    if num_gpus > 1:
        train_sampler = DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **kwargs)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=False, **kwargs)
        
    return dataloader


if __name__ == '__main__':
    # Testing the PDNSDataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Root directory of PDNS')
    parser.add_argument('--synthesized_speakers_csv', help='Path to the synthesized speakers csv file')
    parser.add_argument('--reference_speakers_csv', help='Path to the reference speakers csv file')
    parser.add_argument('--crop_length_sec', type=int, default=0, help='Length of the audio clip')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--sample_rate', type=int, default=41000, help='Sample rate')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs')
    args = parser.parse_args()
    
    trainloader = load_PDNSDataset(root=args.root, synthesized_speakers_csv=args.synthesized_speakers_csv, reference_speakers_csv=args.reference_speakers_csv, crop_length_sec=args.crop_length_sec, batch_size=args.batch_size, sample_rate=args.sample_rate, num_gpus=args.num_gpus)
    
    print(f"Number of steps: {len(trainloader)}")

    for (noisy_audio, speaker_embedding), clean_audio in trainloader: 
        clean_audio = clean_audio.cuda()
        speaker_embedding = speaker_embedding.cuda()
        noisy_audio = noisy_audio.cuda()
        print(f"clean.shape: {clean_audio.shape}")
        print(f"speaker_embedding.shape: {speaker_embedding.shape}")
        print(f"noisy.shape: {noisy_audio.shape}")
        print(f"clean {clean_audio[0][0][0]}")
        print(f"speaker_embedding {speaker_embedding[0][0]}")
        print(f"noisy {noisy_audio[0][0][0]}")
        # print(clean_audio.shape, noisy_audio.shape)
