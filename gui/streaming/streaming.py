from sounddevice import InputStream, OutputStream, query_devices
import torch
import torchaudio
import pickle

import sys
sys.path.append('../rttse')

from streaming.streamer import Streamer, PDenoiserStreamer, PFFSNStreamer
from embedders.TitaNet import TitaNet
from embedders.ECAPATDNN import ECAPATDNN

def parse_audio_device(device):
    if device is None:
        return device
    try:
        return int(device)
    except ValueError:
        return device
    
def load_speaker_embedder(speaker_embedder: str, device: str):
    if "TITANET" in speaker_embedder.upper():
        speaker_embedder = TitaNet()
    elif "ECAPA" in speaker_embedder.upper():
        speaker_embedder = ECAPATDNN(device=device)
    else:
        with open(speaker_embedder, 'rb') as f:
            speaker_embedder = pickle.load(f)
    return speaker_embedder

def setup_model(model_path: str) -> torch.nn.Module:
    return torch.load(model_path)

def initialize_streamer(cfg: dict, reference_audio_path: str) -> Streamer:
    torch.set_num_threads(cfg['settings']['num_threads'])

    device = cfg["settings"]["device"]

    speaker_embedder = load_speaker_embedder(cfg["model"]['speaker_embedder'], device)
    print(f"Speaker embedder loaded from {cfg['model']['speaker_embedder']}")
    
    if isinstance(speaker_embedder, torch.nn.Module):
        speaker_embedder.to(device)
        speaker_embedder.eval()
    
    if cfg["model"]["load_reference_audio"]:
        reference_audio, reference_sr = torchaudio.load(reference_audio_path)
        if reference_sr != cfg['settings']['sr']:
            reference_audio = torchaudio.transforms.Resample(reference_sr, cfg['settings']['sr'])(reference_audio)
        reference_audio = reference_audio.to(device)
        with torch.no_grad():
            embedding = speaker_embedder(reference_audio).to(device)
    else:
        with torch.no_grad():
            embedding = speaker_embedder(reference_audio_path).to(device)
    del speaker_embedder

    model = setup_model(cfg["model"]["speech_enhancer"]).to(device)
    print(f"Model loaded from {cfg['model']['speech_enhancer']}")
    model.eval()

    if "PFFSN" in cfg["model"]["name"]:
        streamer = PFFSNStreamer(model, 
                                 embedding, 
                                 dry=cfg["settings"]["dry"], 
                                 num_frames=cfg["settings"]["num_frames"],
                                 device=device)
    elif "PDenoiser" in cfg["model"]["name"]:
        streamer = PDenoiserStreamer(model, 
                                     embedding, 
                                     dry=cfg["settings"]["dry"], 
                                     num_frames=cfg["settings"]["num_frames"],
                                     device=device)
    print("Streamer initialized.")
    return streamer

def stream_pipeline(streamer: Streamer, cfg: dict, device_in: int, device_out: int):
    caps = query_devices(device_in, "input")
    channels_in = min(caps['max_input_channels'], 2)
    stream_in = InputStream(
        device=device_in,
        samplerate=cfg['settings']['sr'],
        channels=channels_in)
    print("Audio input initialized.")

    caps = query_devices(device_out, "output")
    channels_out = min(caps['max_output_channels'], 2)
    stream_out = OutputStream(
        device=device_out,
        samplerate=cfg['settings']['sr'],
        channels=channels_out)
    print("Audio output initialized.")

    stream_in.start()
    stream_out.start()
    first = True
    current_time = 0
    last_log_time = 0
    last_error_time = 0
    cooldown_time = 2
    log_delta = 10
    device = cfg['settings']['device']
    sr = cfg['settings']['sr']
    sr_ms = sr / 1000
    stride_ms = streamer.stride / sr_ms
    print(f"Ready to process audio, total lag: {streamer.total_length / sr_ms:.1f}ms.")
    while True:
        try:
            if current_time > last_log_time + log_delta:
                last_log_time = current_time
                tpf = streamer.time_per_frame * 1000
                rtf = tpf / stride_ms
                print(f"\rtime per frame: {tpf:.1f}ms, RTF: {rtf:.2f}", end='')
                streamer.reset_time_per_frame()

            length = streamer.total_length if first else streamer.stride
            first = False
            current_time += length / sr
            frame, overflow = stream_in.read(length)
            frame = torch.from_numpy(frame).mean(dim=1).to(device)
            with torch.no_grad():
                out = streamer.feed(frame[None])[0]
            if not out.numel():
                continue
            # if cfg.streaming.compressor:
            #     out = 0.99 * torch.tanh(out)
            out = out[:, None].repeat(1, channels_out)
            mx = out.abs().max().item()
            if mx > 1:
                print("\nClipping!!")
            out.clamp_(-1, 1)
            out = out.cpu().numpy()
            underflow = stream_out.write(out)
            if overflow or underflow:
                if current_time >= last_error_time + cooldown_time:
                    last_error_time = current_time
                    tpf = 1000 * streamer.time_per_frame
                    print(f"\nNot processing audio fast enough, time per frame is {tpf:.1f}ms "
                          f"(should be less than {stride_ms:.1f}ms).")
        except KeyboardInterrupt:
            print("\nStopping")
            break
    stream_out.stop()
    stream_in.stop()