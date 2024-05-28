import tkinter as tk
from tkinter import ttk, filedialog
import os

class ReferenceVoiceComponent:
    padding_val = 10

    def __init__(self, root):
        self.root = root
        self.reference_audio_path = tk.StringVar()
        self.create_upload_button()
        self.create_reference_audio_label()
        self.apply_styles()

    def create_upload_button(self):
        self.upload_button = ttk.Button(self.root, text="Upload Reference Audio", command=self.upload_audio, style="Custom.TButton", padding=int(self.padding_val))
        self.upload_button.place(relx=0.05, rely=0.3, anchor=tk.W)

    def create_reference_audio_label(self):
        self.reference_audio_label = ttk.Label(self.root, text="Nothing selected", style="Custom.TButton", padding=int(self.padding_val))
        self.reference_audio_label.place(relx=0.55, rely=0.3, anchor=tk.W)

    def upload_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.m4a")])
        if file_path:
            self.reference_audio_path.set(file_path)
            print(f"Selected file: {self.reference_audio_path.get()}")
            self.reference_audio_label.configure(text=f"{os.path.basename(file_path)}")

    def apply_styles(self):
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12))
        style.configure("TLabel", font=("Helvetica", 12))
        self.upload_button.configure(style="TButton")
        self.reference_audio_label.configure(style="TLabel")
