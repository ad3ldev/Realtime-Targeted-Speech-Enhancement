import pyaudio
import numpy as np
import torch
import hydra
import time

import keyboard

from archs.WaveTita import WaveTita

mode = 'enhance'

def select_output_device(py_audio):
    output_index = -1
    for i in range(py_audio.get_device_count()):
        if('CABLE Input' in py_audio.get_device_info_by_index(i)['name']):
            print('Virtual cable found at index', i)
            output_index = i
            break
    return output_index

def select_input_device(p):
    print('Available audio devices:')
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if(device_info['maxInputChannels'] > 0 and device_info['hostApi'] == 0):
            print(f"{i}: {device_info['name']}")
    
    print('Please select the index of the input device you want to stream: ')
    input_index = int(input())
    
    if(input_index == -1):
        input_index = p.get_default_input_device_info()['index']
    return input_index

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
    wave_tita = WaveTita(None)
    return wave_tita.speaker_embedder

def setup_model(model_cfg):
    model = hydra.utils.instantiate(model_cfg)
    return model

@hydra.main(version_base=None, config_path="../config", config_name="stream_config")
def stream_pipeline(cfg):
    py_audio = pyaudio.PyAudio()

    # Select the input device
    input_index = select_input_device(py_audio)
    
    # Select the output device (virtual cable)
    output_index = select_output_device(py_audio)
    
    if(output_index == -1):
        print('No virtual cable found')
        print('Please install VB-Audio Virtual Cable and try again')
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")

    model = setup_model(cfg.model).to(device)
    speaker_embedder = load_speaker_embedder()
    
    embedding = speaker_embedder.get_embedding(select_reference_audio()).to(device)
    
    window_length_ms = cfg.streaming.window_size / cfg.streaming.fs * 1000
    
    keyboard.on_press_key('m', lambda _: change_mode())
    
    # Testing
    # import torchaudio
    # test, fs = torchaudio.load("C:/Users/imoha/Downloads/test.wav")
    # if fs != cfg.streaming.fs:
    #     test = torchaudio.transforms.Resample(fs, cfg.streaming.fs)(test)
    # test = test.to(device)
    # # Change number of channels to 1
    # if test.shape[0] > 1:
    #     test = test.mean(dim=0, keepdim=True)
    
    # with torch.no_grad():
    #     test_enhanced = model.net.speech_enhancer(test, embedding)
    # torchaudio.save('test_enhanced.wav', test_enhanced[0].cpu(), cfg.streaming.fs)
    
    # Open a stream with the selected input device as the input
    # and the virtual cable as the output device and stream the audio
    try:
        input_stream = py_audio.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=cfg.streaming.fs,
                        input=True,
                        input_device_index=input_index)
        
        # Open a stream with the virtual cable as the output device
        output_stream = py_audio.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=cfg.streaming.fs,
                        output=True,
                        output_device_index=output_index)
        i = 0
        while input_stream.is_active():
            # Read audio data from the stream
            audio_data = input_stream.read(cfg.streaming.window_size)
            
            # If the mode is original, just stream the audio data
            if mode == 'original':
                output_stream.write(audio_data)
                continue
                        
            # Convert the audio data to a numpy array
            audio_data = np.frombuffer(audio_data, dtype=np.float32)
            audio_data = np.copy(audio_data)
            
            # Convert the numpy array to a PyTorch tensor
            audio_tensor = torch.from_numpy(audio_data).to(device).unsqueeze(0)
                        
            # save the audio tensor to a file
            # torchaudio.save(f'output/{time.time()}.wav', audio_tensor, 16000)
            
            start_time = time.time()
            with torch.no_grad():
                enhanced = model.net.speech_enhancer(audio_tensor, embedding)
            processing_time = (time.time() - start_time) * 1000
            rtf = processing_time / window_length_ms
            
            if i % 100 == 0:
                print(f"\rProcessing time: {processing_time:.2f} ms\tRTF: {rtf:.3f}", end="")            
            
            output_stream.write(enhanced[0].cpu().numpy().tobytes())
            
            i += 1
            
            # Stream the fake audio data
            # output_stream.write(audio_tensor.numpy().tobytes())
    except KeyboardInterrupt:
        pass
    
    # Close the stream and PyAudio
    output_stream.stop_stream()
    output_stream.close()
    input_stream.stop_stream()
    input_stream.close()
    py_audio.terminate()

if __name__ == "__main__":
    stream_pipeline()