import pyaudio
import numpy as np
import torch
import hydra
import time
import multiprocessing
import librosa

# import keyboard

from embedders.TitaNet import TitaNet

mode = 'original'
queue = multiprocessing.Queue()
initialization_lock = multiprocessing.Lock()

def select_output_device(py_audio):
    output_index = -1
    for i in range(py_audio.get_device_count()):
        if "CABLE Input" in py_audio.get_device_info_by_index(i)["name"] or "BlackHole" in py_audio.get_device_info_by_index(i)["name"]:
            output_index = i
            print("Virtual cable found at index", i)
            break

    return output_index 

def select_input_device(p):
    print('Available audio devices:')
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if(device_info['maxInputChannels'] > 0 and device_info['hostApi'] == 0):
            print(f"{i}: {device_info}")
    
    print('Please select the index of the input device you want to stream: ')
    input_index = int(input())
    
    if(input_index == -1):
        input_index = p.get_default_input_device_info()['index']
    return input_index, int(p.get_device_info_by_index(input_index)['defaultSampleRate'])

def select_reference_audio():
    print('Please enter the path to the reference audio file: ')
    reference_audio_path = input()
    return reference_audio_path

def change_mode():
    global mode
    if mode == 'enhance':
        mode = 'original'
    else:
        mode = 'enhance'
    print(f'\nMode changed to {mode}')
    
# TODO: Use TitaNet or separate it using config
def load_speaker_embedder():
    return TitaNet()

def setup_model(model_cfg):
    model = hydra.utils.instantiate(model_cfg)
    return model

def input_streaming(cfg, queue: multiprocessing.Queue, initialization_lock, input_index: int, input_sr: int):
    def input_callback(in_data, frame_count, time_info, status):
        queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    initialization_lock.acquire()
    initialization_lock.release()
    print("Initializing input stream...")
    
    py_audio = pyaudio.PyAudio()
    
    input_frames = cfg.streaming.window_size * input_sr // 1000
    print(f"Input frames: {input_frames}")
    
    print("Starting input stream...")
    input_stream = py_audio.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=input_sr,
                    frames_per_buffer=input_frames,
                    input=True,
                    input_device_index=input_index,
                    stream_callback=input_callback
                    )
    print("Input stream started")
    
    try:
        while input_stream.is_active():
            time.sleep(0.1)
            continue
    except KeyboardInterrupt:
        pass
    
    input_stream.close()

def output_streaming(cfg, queue: multiprocessing.Queue, initialization_lock, output_index: int, reference_audio_path: str, input_sr: int):
    initialization_lock.acquire()

    global mode
        
    print("Initializing output stream...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")

    model = setup_model(cfg.model).to(device)
    speaker_embedder = load_speaker_embedder()
    
    embedding = speaker_embedder.embed(reference_audio_path).to(device)
    
    output_window_length_ms = cfg.streaming.window_size
    i = 0
    
    py_audio = pyaudio.PyAudio()
    
    print("Starting output stream...")
    output_stream = py_audio.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=cfg.streaming.fs,
                    output=True,
                    output_device_index=output_index)
    
    initialization_lock.release()
    
    # Empty the queue before starting
    while not queue.empty():
        queue.get()
    
    print("Starting to stream audio")
    while output_stream.is_active():
        audio_data = queue.get()
        start_time = time.time()
        audio_data = np.frombuffer(audio_data, dtype=np.float32)
        audio_data = np.copy(audio_data)
        audio_data = librosa.resample(audio_data, orig_sr=input_sr, target_sr=cfg.streaming.fs)
        if mode == 'original':
            processing_time = (time.time() - start_time) * 1000
            output_stream.write(audio_data.tobytes())
        else:
            audio_tensor = torch.from_numpy(audio_data).to(device).unsqueeze(0)
            with torch.no_grad():
                enhanced = model.net.speech_enhancer(audio_tensor, embedding)
            processing_time = (time.time() - start_time) * 1000
            output_stream.write(enhanced[0].cpu().numpy().tobytes())
        if i % 100 == 0:
            rtf = processing_time / output_window_length_ms
            print(f"\rProcessing time: {processing_time:.2f} ms\tRTF: {rtf:.3f}, i: {i}, queue.length: {queue.qsize()}", end="")
        i += 1
        
    output_stream.close()

@hydra.main(version_base=None, config_path="../config", config_name="stream_config")
def stream_pipeline(cfg):
    py_audio = pyaudio.PyAudio()

    # Select the input device
    input_index, input_sr = select_input_device(py_audio)
    
    # Select the output device (virtual cable)
    output_index = select_output_device(py_audio)
    
    if(output_index == -1):
        print('No virtual cable found')
        print('Please install VB-Audio Virtual Cable and try again')
        return
    
    processes = []
    processes.append(multiprocessing.Process(target=output_streaming, args=(cfg, queue, initialization_lock, output_index, select_reference_audio(), input_sr)))
    processes.append(multiprocessing.Process(target=input_streaming, args=(cfg, queue, initialization_lock, input_index, input_sr)))
    
    for process in processes:
        process.start()
    
    for process in processes:
        process.join()
    
    py_audio.terminate()


if __name__ == "__main__":
    stream_pipeline()