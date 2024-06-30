from sounddevice import InputStream, OutputStream, query_devices, default
from threading import Event, Thread
import torch
import torchaudio
import pickle

import sys
sys.path.append('../rttse')

from streaming.streamer import Streamer, PDenoiserStreamer, PFFSNStreamer
from embedders.TitaNet import TitaNet
from embedders.ECAPATDNN import ECAPATDNN

def parse_audio_device(device_name, type):
    """
    Parse the audio device name to get the device index.
    """
    devices = query_devices()
    for device in devices:
        print(device)
        if device_name in device['name'] and device['max_' + type + '_channels'] > 0 and device['hostapi'] == default.hostapi:
            return device['index']
    return None
    
def load_speaker_embedder(speaker_embedder: str, device: str):
    if "TITA" in speaker_embedder.upper():
        speaker_embedder = TitaNet()
    elif "ECAPA" in speaker_embedder.upper():
        speaker_embedder = ECAPATDNN(device=device)
    else:
        with open(speaker_embedder, 'rb') as f:
            speaker_embedder = pickle.load(f)
    return speaker_embedder

def compute_speaker_embedding(cfg):
    device = cfg["settings"]["device"]
    speaker_embedder = load_speaker_embedder(cfg["model"]['speaker_embedder'], device)
    print(f"Speaker embedder loaded from {cfg['model']['speaker_embedder']}")
    
    # if isinstance(speaker_embedder, torch.nn.Module):
    #     speaker_embedder.to(device)
    #     speaker_embedder.eval()
    
    if cfg["model"]["load_reference_audio"]:
        reference_audio, reference_sr = torchaudio.load(cfg['reference_audio_path'])
        if reference_sr != cfg['settings']['sr']:
            reference_audio = torchaudio.transforms.Resample(reference_sr, cfg['settings']['sr'])(reference_audio)
        reference_audio = reference_audio.to(device)
        with torch.no_grad():
            embedding = speaker_embedder(reference_audio).to(device)
    else:
        with torch.no_grad():
            embedding = speaker_embedder(cfg['reference_audio_path']).to(device)
    del speaker_embedder
    return embedding

def setup_model(model_path: str) -> torch.nn.Module:
    return torch.load(model_path)

def initialize_streamer(cfg: dict) -> Streamer:
    torch.set_num_threads(cfg['settings']['num_threads'])

    device = cfg["settings"]["device"]

    embedding = compute_speaker_embedding(cfg, device) if 'reference_audio_path' in cfg else None

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

def start_streaming(streamer, cfg, stop_event: Event, finish_event: Event):
    """
    Start the streaming process in a separate thread.
    """
    stop_event.clear()
    finish_event.clear()
    streaming_thread = Thread(target=stream_pipeline, args=(streamer, cfg, stop_event, finish_event))
    streaming_thread.start()
    return streaming_thread

def stop_streaming(stop_event: Event, finish_event: Event):
    """
    Stop the streaming process.
    """
    stop_event.set()
    finish_event.wait()

def stream_pipeline(streamer: Streamer, cfg: dict, stop_event: Event, finish_event: Event):
    print("\n\n\n\n\n\nStarting streaming pipeline\n\n\n\n\n\n")
    
    if streamer.embedding is None:
        print("No reference audio provided, exiting.")
        finish_event.set()
    
    device_in = parse_audio_device(cfg['settings']['in_device'], "input")
    device_out = parse_audio_device(cfg['settings']['out_device'], "output")
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
    while not stop_event.is_set():
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
        except Exception as e:
            print(e)
            break
    try:
        stream_out.stop()
        stream_in.stop()
    except Exception as e:
        print(e)
    
    finish_event.set()