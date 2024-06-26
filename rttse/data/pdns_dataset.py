import os
import numpy as np
import pandas as pd
import re

import torch
import torchaudio

from typing import Literal
from utils.path_utils import get_file_name_without_extension
from torch.utils.data import Dataset


class PDNSCollate:
    def __init__(self):
        pass
    
    def __call__(self, batch):
        clean_audio = torch.stack([data["clean"] for data in batch]) if batch[0]["clean"] is not None else None
        noisy_audio = torch.stack([data["noisy"] for data in batch])
        reference_path = [data["reference_path"] for data in batch]
        noisy_filename = [data["noisy_filename"] for data in batch]
        clean_filename = [data["clean_filename"] for data in batch]
        reference_filename = [data["reference_filename"] for data in batch]
        if "reference" in batch[0]:
            min_length = min([data["reference"].shape[1] for data in batch])
            reference = torch.stack([data["reference"][0, :min_length] for data in batch])
        else:
            reference = None
        index = [data["index"] for data in batch]
        return {
            "clean": clean_audio,
            "noisy": noisy_audio,
            "reference_path": reference_path,
            "reference": reference,
            "index": index,
            "noisy_filename": noisy_filename,
            "clean_filename": clean_filename,
            "reference_filename": reference_filename
        }

class PDNSDataset(Dataset):
    """
    Create a Dataset for the PDNS dataset. The dataset is created by matching the speakers from the synthesized speakers csv file to the speakers csv files.
    """
    
    def __init__(
        self, 
        split: Literal['train', 'val', 'test'],
        clean_paths: list,
        noisy_paths: list,
        synthesized_speakers_paths: list = None,
        dataset_speakers_paths: list = None,
        speaker_reference_paths: list = None,
        sr = 44100, 
        crop_length_sec = 0, 
        mode: Literal['all', 'ps', 'pn', 'psn'] = 'all',
        seed: int = 42,
        reference_tensor: bool = False,
        reference_length_sec: int = 10
        ):
        """ Creates a PDNSDataset object.

        Arguments:
        ----------
        - split (Literal['train', 'val', 'test']): The split of the dataset.
        - clean_paths (list): A list of paths to the clean audio files.
        - noisy_paths (list): A list of paths to the noisy audio files.
        - speaker_reference_paths (list, optional): A list of paths to the speaker reference files. If None, the reference_speakers_csv is used. Defaults to None.            
        - synthesized_speakers_paths (list, optional): A list of csv paths containing the synthesization sources of each generated audio clip. This is used to find the primary speaker of each clip.
        - dataset_speakers_paths (list, optional): A list of csv paths containing the speakers of all raw files from the dataset. This is used to get a clean audio clip related to a sepcific speaker.
        - sr (int, optional): The required sampling rate of all the data. Defaults to 41000.
        - crop_length_sec (int, optional): The required length of each audio clip. If zero doesn't crop. Defaults to 0.
        - mode (Literal[&#39;all&#39;, &#39;ps&#39;, &#39;pn&#39;, &#39;psn&#39;], optional): The noisy speech mode. Used for the selection of noisy audio clips. "all" uses all of the noisy clips, "ps" stands for primary and secondry, "pn" stands for primary and noisy, "psn" stands for primary secondary and noisy. Defaults to 'all'.
        - seed (int, optional): The seed used to select the reference audio clip. Defaults to 42.
        - reference_tensor (bool, optional): Whether the reference needed to be loaded as a tensor or not. Defaults to False.

        Raises:
        -------
        ValueError: In case of an unknown noisy file which doesn't start with 'primary', 'ps', or 'psn'.
        """
        super(PDNSDataset).__init__()

        self.split = split
        self.rng = np.random.default_rng(seed) # May need to change this to torch seed
        
        self.crop_length_sec = crop_length_sec
        self.sr = sr
        self.reference_tensor = reference_tensor
        self.reference_length = reference_length_sec * sr

        noisy_files = [os.path.join(noisy_path, file) for noisy_path in noisy_paths for file in os.listdir(noisy_path)]
        clean_files = [os.path.join(clean_path, file) for clean_path in clean_paths for file in os.listdir(clean_path)] if split != 'test' else [None]*len(noisy_files) # No clean files for test
        
        # Load reference speaker csv
        if speaker_reference_paths is None:
            assert dataset_speakers_paths is not None, "Either speaker_reference_paths or dataset_speakers_paths should be provided"
            reference_speakers = pd.concat([pd.read_csv(reference_speakers_path) for reference_speakers_path in dataset_speakers_paths])
            reference_speakers = reference_speakers[reference_speakers['speaker_type'] == 'primary']
            self.reference_files = dict()
            for _, row in reference_speakers.iterrows():
                if row['speaker_id'] not in self.reference_files:
                    self.reference_files[row['speaker_id']] = [row['filename']]
                else:
                    self.reference_files[row['speaker_id']].append(row['filename'])
            
            assert synthesized_speakers_paths is not None, "speaker_reference_paths or synthesized_speakers_paths should be provided"
            # Load synthesized speaker csv
            synthesized_speakers_df = pd.concat([pd.read_csv(synthesized_speakers_path) for synthesized_speakers_path in synthesized_speakers_paths])
            synthesized_primary_speakers = synthesized_speakers_df['primary_speaker'].tolist()
        else:
            self.reference_files = [os.path.join(speaker_reference_path, file) for speaker_reference_path in speaker_reference_paths for file in os.listdir(speaker_reference_path)]
            self.reference_files.sort()
        
        if split == 'train': # Synthesized
            assert len(clean_files)*3 == len(noisy_files), "Number of clean and noisy files does not match" # 3 noisy files per clean file
            assert len(clean_files) == len(synthesized_primary_speakers), "Number of clean files and synthesized primary speakers does not match"
        elif split == 'val':
            assert len(clean_files) == len(noisy_files), "Number of clean and noisy files does not match"
            assert len(clean_files) == len(self.reference_files), "Number of clean and reference files does not match"
        elif split == 'test':
            assert len(noisy_files) == len(self.reference_files), "Number of noisy and reference files does not match"
        
        if split != 'test':
            clean_files = self._sort_files(clean_files)
        noisy_files = self._sort_files(noisy_files)
        
        if split == 'train':
            noisy_files = self.choose_train_noisy_files(mode, noisy_files)
            if(len(clean_files)*3 == len(noisy_files)): # In case of 'all' mode, repeat the noisy files 3 times
                clean_files = clean_files*3
                synthesized_primary_speakers = synthesized_primary_speakers*3
        
        # Number of clean and noisy files should match at this point
        assert len(clean_files) == len(noisy_files), "Number of clean and noisy files does not match" 
        
        # Create a list of tuples of clean files, noisy files and primary speakers
        if speaker_reference_paths is None:
            self.files = list(zip(clean_files, noisy_files, synthesized_primary_speakers))
        else:
            self.files = list(zip(clean_files, noisy_files, self.reference_files))

    def choose_train_noisy_files(self, mode, noisy_files):
        noisy_ps_files = []
        noisy_pn_files = []
        noisy_psn_files = []
        for noisy_file in noisy_files:
            noisy_file_basename = os.path.basename(noisy_file)
            if noisy_file_basename.startswith('primary'):
                noisy_pn_files.append(noisy_file)
            elif noisy_file_basename.startswith('psn'):
                noisy_psn_files.append(noisy_file)
            elif noisy_file_basename.startswith('ps'):
                noisy_ps_files.append(noisy_file)
            else:
                raise ValueError(f"Unknown noise type for file {noisy_file}")
        
        noisy_ps_files = self._sort_files(noisy_ps_files)
        noisy_pn_files = self._sort_files(noisy_pn_files)
        noisy_psn_files = self._sort_files(noisy_psn_files)
        noisy_files = noisy_ps_files + noisy_pn_files + noisy_psn_files
        
        if mode == 'ps':
            noisy_files = noisy_ps_files
        elif mode == 'pn':
            noisy_files = noisy_pn_files
        elif mode == 'psn':
            noisy_files = noisy_psn_files
        return noisy_files

    def _sort_files(self, files):
        pattern = r'fileid_(\d+)'
        if(len(files) >= 1 and re.search(pattern, files[0]) is None):
            return sorted(files)
        return sorted(files, key=lambda x: int(re.search(pattern, x).group(1)))

    def __getitem__(self, n):
        file = self.files[n]
        noisy_audio, noisy_sr = torchaudio.load(file[1])
        
        # Resample the audio to the desired sample rate
        if noisy_sr != self.sr:
            noisy_audio = torchaudio.transforms.Resample(orig_freq=noisy_sr, new_freq=self.sr)(noisy_audio)
        noisy_audio = noisy_audio.squeeze(0)
        
        crop_length = int(self.crop_length_sec * self.sr)
        assert crop_length < len(noisy_audio), f"Crop length {crop_length} is greater than the length of the audio {len(noisy_audio)}"

        # Random crop
        if crop_length > 0:
            start = np.random.randint(low=0, high=len(noisy_audio) - crop_length + 1)
            noisy_audio = noisy_audio[start:(start + crop_length)]
                
        # Load clean audio if it exists
        if file[0] is not None:
            clean_audio, clean_sr = torchaudio.load(file[0])
            if clean_sr != self.sr:
                clean_audio = torchaudio.transforms.Resample(orig_freq=clean_sr, new_freq=self.sr)(clean_audio)
            clean_audio = clean_audio.squeeze(0)
            if crop_length > 0:
                clean_audio = clean_audio[start:(start + crop_length)]
            assert len(clean_audio) == len(noisy_audio), "Length of clean: " + file[0] + " and noisy audio: " + file[1] + " does not match"
            clean_audio = clean_audio.unsqueeze(0)
            clean_filename = get_file_name_without_extension(file[0])
        else:
            clean_audio, clean_sr = None, None 
            clean_filename = None
        
        noisy_audio = noisy_audio.unsqueeze(0)

        # Select a random speaker from the clean speakers
        if isinstance(self.reference_files, dict):
            reference_file = self.rng.choice(self.reference_files[file[2]])
        # The reference files are already loaded as a list, so just select the indexed file
        elif isinstance(self.reference_files, list):
            reference_file = self.reference_files[n]
        else:
            raise ValueError("Unknown reference file type")
                
        data = {
            "clean": clean_audio,
            "noisy": noisy_audio,
            "reference_path": reference_file,
            "index": n,
            "clean_filename": clean_filename,
            "noisy_filename": get_file_name_without_extension(file[1]),
            "reference_filename": get_file_name_without_extension(file[2]) 
        }
        
        # Load reference audio if reference_tensor is True
        if self.reference_tensor:
            reference_audio, reference_sr = torchaudio.load(reference_file)
            if reference_sr != self.sr:
                reference_audio = torchaudio.transforms.Resample(orig_freq=reference_sr, new_freq=self.sr)(reference_audio)
            if self.reference_length == 0:
                pass
            elif reference_audio.shape[1] > self.reference_length:
                reference_audio = reference_audio[:, :self.reference_length]
            elif reference_audio.shape[1] < self.reference_length:
                reference_audio = torch.cat([reference_audio]*int(np.ceil(self.reference_length/reference_audio.shape[1])), dim=1)
            
            data["reference"] = reference_audio

        return data

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


# def load_PDNSDataset(split, clean_paths, noisy_paths, dataset_speakers_paths, synthesized_speakers_paths, speaker_reference_paths, crop_length_sec, batch_size, sample_rate, num_gpus=1):
#     """
#     Get dataloader with distributed sampling
#     """
#     dataset = PDNSDataset(split=split, clean_paths=clean_paths, noisy_paths=noisy_paths, synthesized_speakers_paths=synthesized_speakers_paths, crop_length_sec=crop_length_sec, dataset_speakers_paths=dataset_speakers_paths, speaker_reference_paths=speaker_reference_paths, sr=sample_rate)                                                       
#     kwargs = {"batch_size": batch_size, "num_workers": 4, "pin_memory": False, "drop_last": False, "collate_fn": PDNSCollate()}

#     if num_gpus > 1:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
#         dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **kwargs)
#     else:
#         train_sampler = torch.utils.data.RandomSampler(dataset)
#         dataloader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=False, **kwargs)
        
#     return dataloader


# if __name__ == '__main__':
#     # Testing the PDNSDataset
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--split', help='Split of the dataset')
#     parser.add_argument('--clean_paths', help='Path to the clean audio files', nargs='+')
#     parser.add_argument('--noisy_paths', help='Path to the noisy audio files', nargs='+')
#     parser.add_argument('--dataset_speakers_paths', help='Path to the synthesized speakers csv file', nargs='+')
#     parser.add_argument('--speaker_reference_paths', help='Path to the reference speakers csv file', nargs='+')
#     parser.add_argument('--crop_length_sec', type=int, default=0, help='Length of the audio clip')
#     parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
#     parser.add_argument('--sample_rate', type=int, default=41000, help='Sample rate')
#     parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs')
#     args = parser.parse_args()
    
#     print(args)
    
#     trainloader = load_PDNSDataset(split=args.split, clean_paths=args.clean_paths, noisy_paths=args.noisy_paths, dataset_speakers_paths=args.dataset_speakers_paths, speaker_reference_paths=args.speaker_reference_paths, crop_length_sec=args.crop_length_sec, batch_size=args.batch_size, sample_rate=args.sample_rate, num_gpus=args.num_gpus)
    
#     print(f"Number of steps: {len(trainloader)}")

#     for data in trainloader: 
#         clean_audio = data["clean"]
#         noisy_audio = data["noisy"]
#         reference_path = data["reference_path"]
#         # clean_audio = clean_audio.cuda()
#         # noisy_audio = noisy_audio.cuda()
#         print(f"clean {clean_audio[0][0][0]}")
#         print(f"noisy {noisy_audio[0][0][0]}")
#         print(clean_audio.shape, noisy_audio.shape)
#         print(reference_path)  