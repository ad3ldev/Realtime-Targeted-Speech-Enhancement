from torch import Tensor

import torch
import torch.optim as optim
import os
import argparse

from nemo.collections.asr.models import EncDecSpeakerLabelModel

from Waveformer import Net, loss

from pdns_dataset import load_PDNSDataset

class WaveTita(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(WaveTita, self).__init__(*args, **kwargs)
        self.speaker_embedder = self.load_speaker_embedder()
        network_config = {
            "label_len": 192,
            "L": 32,
            "enc_dim": 512,
            "num_enc_layers": 10,
            "dec_dim": 256,
            "num_dec_layers": 1,
            "dec_buf_len": 13,
            "dec_chunk_size": 13,
            "out_buf_len": 4, 
            "use_pos_enc": "true"
        }
        self.speech_enhancer = self.load_speech_enhancer(network_config)
        self.device = self.detect_device()
    
    def detect_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_speaker_embedder(self) -> None:
        speaker_embedder = EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large').eval()
        # Freeze the speaker embedder
        for param in speaker_embedder.parameters():
            param.requires_grad = False
        return speaker_embedder
    
    def load_speech_enhancer(self, network_config) -> None:
        speech_enhancer = Net(**network_config)
        model_filename = "Waveformer_ckpt.pt"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_filename)
        # Load the weights
        state_dict = torch.load(model_path,
                                 map_location=self.device)["model_state_dict"]
        state_dict.pop("label_embedding.0.weight")
        speech_enhancer.load_state_dict(state_dict, strict=False)
        return speech_enhancer
    
    def forward(self, x, label) -> Tensor:
        return self.speech_enhancer(x, label)
    

if __name__ == "__main__":
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

    # Hyperparameters
    learning_rate = 0.001
    epochs = 10
    batch_size = args.batch_size
    sample_rate = args.sample_rate
    
    # Load model, dataset, and optimizer
    model = WaveTita()
    model = model.to(model.device)  # Move model to CUDA if available
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(epochs):
        for i, data in enumerate(trainloader):
            noisy_audio = data["noisy"].to(model.device)
            reference_path = data["reference_path"]
            labels = torch.zeros((batch_size, 1, 192))

            for i in range(batch_size):
                labels[i] = model.speaker_embedder.get_speaker_embeddings(reference_path[i])

            labels = labels.to(model.device)
            # Forward pass
            enhanced_audio = model(noisy_audio, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress
            if i % 10 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")