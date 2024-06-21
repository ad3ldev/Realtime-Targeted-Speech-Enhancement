import json
import os
import random
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import numpy as np

class YTData(torch.utils.data.Dataset):
    def __init__(self, data_manifest, data_root, sr=16000, crop_length_sec=None, mix_levels=(0.667, 0.444, 0.296, 0.1), take=None):
        self.data = data_manifest
        self.data_root = data_root
        self.sr = sr
        self.crop_length_sec = crop_length_sec
        self.mix_levels = mix_levels

        with open(data_manifest, "r") as f:
            self.data = json.load(f)
            if take:
                self.data = self.data[:take]

    def get_data_path(self, path):
        return os.path.join(self.data_root, path)


    def load_audio(self, path, normalize=True, crop=False):
        audio, sr = torchaudio.load(self.get_data_path(path), normalize=normalize)
        audio = audio.squeeze()
        if sr != self.sr:
            audio = T.Resample(sr, self.sr, dtype=audio.dtype)(audio)
            # audio = F.resample(audio, sr, self.sr, dtype=audio.dtype)
        
        if crop and self.crop_length_sec:
            crop_length = int(self.crop_length_sec * self.sr)
            assert crop_length < len(audio), path

            # Random crop
            if crop_length > 0:
                start = np.random.randint(low=0, high=len(audio) - crop_length + 1)
                audio = audio[start:(start + crop_length)]
                assert len(audio) == crop_length

        return audio

    def generate_sample(self, data_record):
        ## load the 3 audio sample
        ## mix AClean, BClean with one of 3 ratios (0.667, 0.444, 0.296)
        ## return ARef, mixed, AClean
        aRef   = self.load_audio(data_record['speakerAReference'])
        aClean = self.load_audio(data_record['speakerAClean'], crop=True)
        bClean = self.load_audio(data_record['speakerBClean'], crop=True)


        mix_level = random.choices(self.mix_levels, k=1)[0]

        mixed = aClean + mix_level * bClean
        
        print(mixed.shape)

        return aRef, mixed, aClean

    
    def __getitem__(self, idx):
        sample = self.generate_sample(self.data[idx])
        data = {
            "clean": sample[2],
            "noisy": sample[1],
            "reference": sample[0],
            "index": idx
        }
        return data

    def __len__(self):
        return len(self.data)