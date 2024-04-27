import argparse
import pandas as pd
from tqdm import tqdm

class SpeakerMatcher:
    def __init__(self, speakers_paths: list) -> None:
        """Creates a SpeakerMatcher object. Matches speakers from the synthetic sources csv file to the speakers csv files.

        Args:
            speakers_paths (list): List of paths to the speakers csv files
        """
        
        self.speakers_paths = dict()
        speakers_df = pd.concat([pd.read_csv(speaker_path) for speaker_path in speakers_paths])
        for _, row in tqdm(speakers_df.iterrows(), "Loading speakers"):
            self.speakers_paths[row['filename']] = row['speaker_id']
        print(f"Loaded {len(self.speakers_paths)} speakers")
    
    def match(self, synth_source_path: str) -> pd.DataFrame:
        """Matches speakers from the synthetic sources csv file to the speakers csv files.

        Args:
            synth_source_path (str): Path to the synthetic sources csv file containing the primary and secondary speakers

        Returns:
            pd.DataFrame: DataFrame containing the file index, primary speaker, and secondary speaker
        """
        synth_sources = pd.read_csv(synth_source_path, header=None)
        synth_sources.columns = ['file_idx', 'primary_file', 'secondary_file', 'noise_file']  # file_idx is the index of the synthesized file
        
        matches = []
        for _, row in tqdm(synth_sources.iterrows(), "Matching speakers"):
            primary_speaker = self.speakers_paths[row['primary_file']]
            secondary_speaker = self.speakers_paths[row['secondary_file']]
            matches.append([row['file_idx'], primary_speaker, secondary_speaker])
        
        print(f"Matched {len(matches)} speakers")
        return pd.DataFrame(matches, columns=['file_idx', 'primary_speaker', 'secondary_speaker'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speakers_paths', nargs='+', help='Paths to the speakers csv files comma separated')
    parser.add_argument('--synth_source_path', help='Path to the synthetic sources csv file')
    parser.add_argument('--output_path', help='Path to the output csv file')
    args = parser.parse_args()
    
    speaker_matcher = SpeakerMatcher(args.speakers_paths)
    matches = speaker_matcher.match(args.synth_source_path)
    matches.to_csv(args.output_path, index=False)