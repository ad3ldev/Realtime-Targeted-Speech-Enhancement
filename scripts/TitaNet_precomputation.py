import argparse
import pandas as pd
import torch

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from tqdm import tqdm

def load_speaker_embedder():
    speaker_embedder = EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large').eval()
    # Freeze the speaker embedder
    for param in speaker_embedder.parameters():
        param.requires_grad = False
    return speaker_embedder

def compute_embeddings(reference_paths, start_index = 0):
    speaker_embedder = load_speaker_embedder()
    embeddings = torch.zeros((len(reference_paths), 192))
    embeddings_sources = pd.DataFrame(reference_paths, columns=['source_path'])
    # add an empty column to store the index
    embeddings_sources['index'] = 0
    for i in tqdm(range(len(reference_paths))):
        embeddings[i] = speaker_embedder.get_embedding(reference_paths[i])
        # set the index as i where the reference path is the same as the source path
        embeddings_sources.loc[embeddings_sources['source_path'] == reference_paths[i], 'index'] = i
    return embeddings, embeddings_sources

def save_embeddings(embeddings: torch.Tensor, path):
    torch.save(embeddings.cpu(), path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_paths_csv', type=str, help='Path to the CSV file containing the reference paths')
    parser.add_argument('--output_embeddings', type=str, default='embeddings.pt', help='Path to the output file')
    parser.add_argument('--output_sources', type=str, default='embeddings_sources.csv', help='Path to the output file containing the sources')
    args = parser.parse_args()
    
    reference_paths = pd.read_csv(args.reference_paths_csv)['filename'].tolist()
    embeddings, embeddings_sources = compute_embeddings(reference_paths)
    save_embeddings(embeddings, args.output_embeddings)
    embeddings_sources.to_csv(args.output_sources, index=False)

if __name__ == '__main__':
    main()