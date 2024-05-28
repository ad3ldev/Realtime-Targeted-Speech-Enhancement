import tkinter as tk
from tkinter import ttk
import customtkinter as ctk # Install 'customtkinter' library
from tkinter import filedialog
import os
from sounddevice import query_devices, default
import speech_recognition as sr # Install 'SpeechRecognition' library

# from PIL import Image, ImageTk # Install 'pillow' library
# from playsound import playsound # Install 'playsound' library
# from winsound import SND_FILENAME
# from pygame import mixer # Install 'pygame' library


def get_current_value():
    return '{:.0f}%'.format(dry_set_slider_value.get())

def slider_changed(event):
    new_value = round_to_nearest_multiple_of_step(int(get_current_value()[:-1]), 5)
    current_value_label.configure(text=str(new_value) + '%') # change the 5 for different step
    dry_set_slider_value.set(new_value)

def round_to_nearest_multiple_of_step(value, step):
    return round(value / step) * step


# Create the root window
root = ctk.CTk()
ctk.set_appearance_mode("light")
# ctk.set_default_color_theme("blue")

root.title("Denoiser")
root.geometry("400x300")

# All Tkinter text ##########################
custom_style = ttk.Style()
custom_style.configure("Custom.TButton", font=("Helvetica", 16))  # Set the font size


##############################################

# SWITCH Component (Cancel Noise) ###############
def switch_event():
    switch_state = switch_var.get()
    if switch_state == "on":
        slider.config(state=tk.NORMAL)
        print("Noise Cancelation is on")

    else:
        dry_set_slider_value.set(0)
        current_value_label.configure(text='0%')
        slider.config(state=tk.DISABLED)
        print("Noise Cancelation is off")

switch_var = ctk.StringVar(value="on")
switch_state = switch_var.get()
switch = ctk.CTkSwitch(root, text="Cancel Noise", command=switch_event, variable=switch_var, onvalue="on", offvalue="off")
switch.place(relx=0.95, rely=0.05, anchor='ne')


# SLIDER Component (Dry/Wet) ##########################
dry_set_slider_value = tk.DoubleVar()

sliderFrame = tk.Frame(root)
sliderFrame.columnconfigure(0, weight=1)
sliderFrame.columnconfigure(1, weight=10)
sliderFrame.columnconfigure(2, weight=1)

left_label = ttk.Label(sliderFrame, text="0% (Dry)")
left_label.grid(row=0, column=0, sticky=tk.W)

slider = ttk.Scale(sliderFrame, from_=0, to=100, orient="horizontal", 
                   variable=dry_set_slider_value, command=slider_changed)
slider.grid(row=0, column=1, sticky=tk.W+tk.E)

right_label = ttk.Label(sliderFrame, text="100% (Wet)")
right_label.grid(row=0, column=2, sticky=tk.E)

current_value_label = ttk.Label(sliderFrame, text="Value: 0%")
current_value_label.grid(column=1, row=1, sticky=tk.S)
slider_changed(None)  # Initialize the label with the default value

sliderFrame.pack(fill='x', side="bottom")

##############################################

# Upload reference Component ##########################
def upload_audio():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.m4a")])
    if file_path:
        reference_audio_path.set(file_path)
        print(f"Selected file: {reference_audio_path.get()}")
        reference_audio_label.configure(text=f"{os.path.basename(file_path)}")

reference_audio_path = tk.StringVar()
upload_button = ttk.Button(root, text="Upload Reference Audio", command=upload_audio, style="Custom.TButton", padding=10)
upload_button.place(relx=0.5, rely=0.25, anchor=tk.CENTER)

reference_audio_label = ttk.Label(root, text="Nothing selected", style="Custom.TButton", padding=10)
reference_audio_label.place(relx=0.04, rely=0.4, relwidth=0.9, anchor=tk.W)

##############################################

# Dropdown Component (Select Input Device) ##########################

def get_input_devices():
    devices = query_devices()
    return [device["name"] for device in devices]

def get_default_input_device():
    return default.device

input_devices = get_input_devices()
default_input_device = get_default_input_device()
print(f"Default Device: {default_input_device}")

input_device_var = ctk.StringVar(value=default_input_device)

input_device_label = ttk.Label(root, text="Select Input Device", padding=10)

input_device_dropdown = ttk.Combobox(root, values=input_devices, textvariable=input_device_var, state="readonly")

input_device_label.place(relx=0.05, anchor='nw')

input_device_dropdown.place(relx=0.05, rely=0.1, anchor='nw')

# print(sr.Microphone.list_microphone_names()) 

##############################################


# Play/Stop button to run a sample #################
# def play_reference_audio(audio_path):
#     playsound(audio_path, SND_FILENAME)
#     print("play button pressed!")



# def play_reference_audio(audio_path):
#     try:
#         mixer.init()  # Initialize the mixer
#         sound = mixer.Sound(audio_path)  # Load the audio file
#         sound.play()
#     except Exception as e:
#         print(f"Error playing audio: {e}")


# # Create a CTkButton with default theme
# play_icon = Image.open("play.png")
# play_icon = play_icon.resize((28, 28), Image.LANCZOS)
# play_icon = ctk.CTkImage(play_icon)

# # Customize the button
# button = ctk.CTkButton(
#     master=root,
#     width=32,
#     height=32,
#     border_width=0,
#     corner_radius=8,
#     image=play_icon,
#     command=lambda: play_reference_audio(reference_audio_path.get()),
#     text="",
# )
# button.place(relx=0.95, rely=0.4, anchor=tk.E)

##############################################










root.mainloop()
