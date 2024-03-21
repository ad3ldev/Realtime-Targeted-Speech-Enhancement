from metrics.dnsmos_local import DNSMOSScore
from data.cleanunet_dataset import CleanNoisyPairDataset, load_CleanNoisyPairDataset
from tqdm import tqdm

# Instantiate the metric
dns_mos_metric = DNSMOSScore(input_sampling_rate=16000)

trainloader = load_CleanNoisyPairDataset(
        root = "D:\\Graduation Project\\datasets\\dns",
        subset = "training",
        crop_length_sec = 0,
        batch_size=1,
        sample_rate=16000,
        num_gpus=1
        )


# Loop through your dataset and calculate MOS scores for each batch
for clean_audio, noisy_audio in tqdm(trainloader): 
    for i in range(len(clean_audio)):
        dns_mos_metric.update(noisy_audio[i].squeeze(), clean_audio[i].squeeze())


# Compute final MOS scores
mos_scores = dns_mos_metric.compute()
print("MOS Scores:", mos_scores)
