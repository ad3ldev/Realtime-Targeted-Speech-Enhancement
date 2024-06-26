from functools import reduce
import random
import os
import json
from sklearn.model_selection import train_test_split


data_path = "/kaggle/input/arabic-clean-audio-dataset/private-data"

dirs = list(map(lambda x: os.path.join(data_path, x), os.listdir(data_path)))[:-1]
dirs = os.listdir(data_path)[:-1]

train_dirs = list(filter(lambda x: not (((int(x) - 1) % 12) in {10, 11}), dirs))

sorted(train_dirs)


def get_data_path(dir):
    return os.path.join(data_path, dir)

def build_manifest(speaker_dir, others, noise_count=10):
    others_audio = [[os.path.join(other, file) for file in os.listdir(get_data_path(other))] for other in others]
    others_audio = list(reduce(lambda x, y: x + y, others_audio, []))
    result = []
    ref_samples = os.listdir(get_data_path(speaker_dir))
    clean_samples = os.listdir(get_data_path(speaker_dir))
    clean_samples.append("")
    for i in range(min(10, len(ref_samples))):
        for j in range(min(11, len(clean_samples))):
            if i == j: continue
            noises = random.choices(others_audio, k=noise_count)
            for noise in noises:
                result.append({
                    "speakerAReference": os.path.join(speaker_dir, ref_samples[i]),
                    "speakerAClean": os.path.join(speaker_dir, clean_samples[j]),
                    "speakerBClean": noise
                })

    return result


if __name__ == "__main__":
    train_data = []
    for speaker_dir in train_dirs:
        others = set(train_dirs)
        others.remove(speaker_dir)
        train_data.extend(build_manifest(speaker_dir, others))

    train_data, val_data = train_test_split(train_data, test_size=0.1, shuffle=True, random_state=42)
    print(train_data[-1], len(train_data))

    with open("train_data_manifest.json", "w+") as fp:
        json.dump(train_data, fp)

    with open("val_data_manifest.json", "w+") as fp:
        json.dump(val_data, fp)