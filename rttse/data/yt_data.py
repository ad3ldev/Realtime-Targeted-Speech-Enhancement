import json
import os
import random
import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.functional as Fa
import torchaudio.transforms as T

import numpy as np

from utils.logger import get_root_logger

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
    def __init__(self):
        pass

    def __call__(self, sample, idx):
        return sample[2], sample[1], sample[0], idx
    
class DictTransform:
    def __init__(self):
        pass

    def __call__(self, sample, idx):
        get_root_logger().info(f"Clean audio shape (before unsqueeze(1)) inside DictTransform: {sample[2].shape}")
        get_root_logger().info(f"Noisy audio shape (before unsqueeze(1)) inside DictTransform: {sample[1].shape}")
        get_root_logger().info(f"Reference audio shape (before unsqueeze(1)) inside DictTransform: {sample[0].shape}")
        return {
            "clean": sample[2].unsqueeze(1),
            "noisy": sample[1].unsqueeze(1),
            "reference_path": sample[3],
            "reference": sample[0],
            "index": idx
        }
    
def pad_to_length(audio, length):
    if len(audio) < length:
        return F.pad(audio, (0, length - len(audio)))

class YTData(torch.utils.data.Dataset):
    def __init__(self, 
                 data_manifest, 
                 data_root, 
                 output_mapper, 
                 sr=16000, 
                 length_sec=None, 
                 reference_length_sec=10, 
                 snrs_db=(20, 15, 10, 6, 3, 0),
                 pitch_shifts=None,
                 take=None):
        self.data = data_manifest
        self.data_root = data_root
        self.sr = sr
        self.length_sec = length_sec
        self.reference_length_sec = reference_length_sec
        self.snrs_db = snrs_db
        self.output_mapper = output_mapper
        self.pitch_shifts = pitch_shifts

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
        aRef   = self.load_audio(data_record['speakerAReference'], length_sec=self.reference_length_sec).unsqueeze(0)
        
        bClean = self.load_audio(data_record['speakerBClean'], length_sec=self.length_sec).unsqueeze(0)

        a_shift = None

        if self.pitch_shifts is not None:
            a_shift = T.PitchShift(self.sr, random.choices(self.pitch_shifts, k=1)[0])
            aRef = a_shift(aRef)
            bClean = T.PitchShift(self.sr, random.choices(self.pitch_shifts, k=1)[0])(bClean)
        
        mix_level = random.choices(self.snrs_db, k=1)

        if data_record['speakerAClean'].endswith('.wav'):
            aClean = self.load_audio(data_record['speakerAClean'], length_sec=self.length_sec).unsqueeze(0)
            if a_shift is not None:
                aClean = a_shift(aClean)
            mixed = Fa.add_noise(aClean, bClean, snr=torch.Tensor(mix_level))
        else:
            aClean = 0 * bClean
            mixed = T.Vol(gain= -mix_level[0] - T.Loudness(sample_rate=self.sr)(bClean.unsqueeze(0)), gain_type="db")(bClean)
            

        return aRef, mixed, aClean, self.get_data_path(data_record['speakerAReference'])

    
    def __getitem__(self, idx):
        sample = self.generate_sample(self.data[idx])
        return self.output_mapper(sample, idx)

    def __len__(self):
        return len(self.data)
    