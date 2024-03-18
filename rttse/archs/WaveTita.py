from torch import Tensor

import torch

from nemo.collections.asr.models import EncDecSpeakerLabelModel

class WaveTita(torch.nn.Module):
    def __init__(self, speech_enhancer, *args, **kwargs) -> None:
        super(WaveTita, self).__init__(*args, **kwargs)
        self.device = self.detect_device()
        self.speaker_embedder = self.load_speaker_embedder()
        self.speech_enhancer = speech_enhancer
    
    def detect_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_speaker_embedder(self) -> None:
        speaker_embedder = EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large').eval()
        # Freeze the speaker embedder
        for param in speaker_embedder.parameters():
            param.requires_grad = False
        return speaker_embedder
    
    def forward(self, data) -> Tensor:
        noisy_audio = data["noisy"].to(self.device)
        reference_path = data["reference_path"]
        batch_size = noisy_audio.shape[0]

        labels = torch.zeros((batch_size, 192))
        for j in range(batch_size):
            labels[j] = self.speaker_embedder.get_embedding(reference_path[j])

        labels = labels.to(self.device)
        return self.speech_enhancer(noisy_audio, labels)
    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--root', help='Root directory of PDNS')
#     parser.add_argument('--synthesized_speakers_csv', help='Path to the synthesized speakers csv file')
#     parser.add_argument('--reference_speakers_csv', help='Path to the reference speakers csv file')
#     parser.add_argument('--crop_length_sec', type=int, default=0, help='Length of the audio clip')
#     parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
#     parser.add_argument('--sample_rate', type=int, default=41000, help='Sample rate')
#     parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs')
#     args = parser.parse_args()
    
#     trainloader = load_PDNSDataset(root=args.root, synthesized_speakers_csv=args.synthesized_speakers_csv, reference_speakers_csv=args.reference_speakers_csv, crop_length_sec=args.crop_length_sec, batch_size=args.batch_size, sample_rate=args.sample_rate, num_gpus=args.num_gpus)

#     # Hyperparameters
#     learning_rate = 0.001
#     epochs = 10
#     batch_size = args.batch_size
#     sample_rate = args.sample_rate
    
#     # Load model, dataset, and optimizer
#     model = WaveTita()
#     model = model.to(model.device)  # Move model to CUDA if available
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)


#     for epoch in range(epochs):
#         for i, data in enumerate(trainloader):
#             noisy_audio = data["noisy"].to(model.device)
#             reference_path = data["reference_path"]
#             clean_audio = data["clean"].to(model.device)
#             labels = torch.zeros((batch_size, 192))

#             for j in range(batch_size):
#                 labels[j] = model.speaker_embedder.get_embedding(reference_path[j])

#             labels = labels.to(model.device)

#             print(f"Shape of noisy audio: {noisy_audio.shape}")
#             print(f"Shape of labels: {labels.shape}")

#             # Forward pass
#             enhanced_audio = model(noisy_audio, labels)

#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss_value = loss(enhanced_audio, clean_audio)
#             loss_value.backward()
#             optimizer.step()

#             # Print progress
#             if i % 10 == 0:
#                 print(f"Epoch {epoch}, Step {i}, Loss: {loss_value.item()}")