import json
import os
import random
import torch
import torchaudio
import torchaudio.functional as F
import torch.nn.functional as F_
import torchaudio.transforms as T

import numpy as np

class YTCollate:
    def __init__(self):
        pass
    
    def __call__(self, batch):
        clean_audio = torch.stack([data["clean"] for data in batch])
        noisy_audio = torch.stack([data["noisy"] for data in batch])
        reference = torch.stack([data["reference"] for data in batch])
        index = [data["index"] for data in batch]
        return {
            "clean": clean_audio,
            "noisy": noisy_audio,
            "reference": reference,
            "index": index
        }
    
class TupleTransform:
    def __call__(self, sample, idx):
        return sample[2], sample[1], sample[0], idx
    
class DictTransform:
    def __call__(self, sample, idx):
        return {
            "clean": sample[2],
            "noisy": sample[1],
            "reference": sample[0],
            "index": idx
        }
    
def pad_to_length(audio, length):
    if len(audio) < length:
        return F_.pad(audio, (0, length - len(audio)))

class YTData(torch.utils.data.Dataset):
    def __init__(self, data_manifest, data_root, output_mapper, sr=16000, length_sec=None, reference_length_sec=10, mix_levels=(0.667, 0.444, 0.296, 0.1), take=None):
        self.data = data_manifest
        self.data_root = data_root
        self.sr = sr
        self.length_sec = length_sec
        self.reference_length_sec = reference_length_sec
        self.mix_levels = mix_levels
        self.output_mapper = output_mapper

        with open(data_manifest, "r") as f:
            self.data = json.load(f)
            if take:
                self.data = self.data[:take]

    def get_data_path(self, path):
        return os.path.join(self.data_root, path)


    def load_audio(self, path, normalize=True, length_sec=None):
        audio, sr = torchaudio.load(self.get_data_path(path), normalize=normalize)
        audio = audio.squeeze()
        if sr != self.sr:
            audio = T.Resample(sr, self.sr, dtype=audio.dtype)(audio)
            # audio = F.resample(audio, sr, self.sr, dtype=audio.dtype)
        
        if length_sec is not None:
            crop_length = int(length_sec * self.sr)
            if crop_length > len(audio):
                audio = pad_to_length(audio, crop_length)
            else:
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
        aRef   = self.load_audio(data_record['speakerAReference'], length_sec=self.reference_length_sec)

        bClean = self.load_audio(data_record['speakerBClean'], length_sec=self.length_sec)

        if data_record['speakerAClean'].endswith('.wav'):
            aClean = self.load_audio(data_record['speakerAClean'], length_sec=self.length_sec)
        else:
            aClean = 0 * bClean


        mix_level = random.choices(self.mix_levels, k=1)[0]

        mixed = aClean + mix_level * bClean
        
        # print(mixed.shape)

        return aRef, mixed, aClean

    
    def __getitem__(self, idx):
        sample = self.generate_sample(self.data[idx])
        # data = {
        #     "clean": sample[2],
        #     "noisy": sample[1],
        #     "reference": sample[0],
        #     "index": idx
        # }
        # print("sample shape:", sample[0].shape, sample[1].shape, sample[2].shape)
        return self.output_mapper(sample, idx)

    def __len__(self):
        return len(self.data)