from tkinter import ttk
import tkinter as tk
import customtkinter as ctk
from dao import create_connection, create_table, insert_reference_audio
from components import InputDeviceComponent, NoiseSuppressionComponent, ReferenceVoiceComponent

class MainApplication:
    resize_after_id = None

    def __init__(self, root):
        self.root = root
        self.root.title("Denoiser")
        self.root.geometry("350x250")
        self.root.bind("<Configure>", self.on_resize)

        db_file = "reference_audios.db"
        user_id = "default"

        # Initialize components
        self.noise_suppression_component = NoiseSuppressionComponent(root)
        self.reference_voice_component = ReferenceVoiceComponent(root, db_file, user_id)
        self.input_device_component = InputDeviceComponent(root)


        self.apply_initial_styles()


    def apply_initial_styles(self, font_size=12):
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", font_size))
        style.configure("TButton", font=("Helvetica", font_size))
        style.configure("TScale", font=("Helvetica", font_size))
        style.configure("TCombobox", font=("Helvetica", font_size))
        self.noise_suppression_component.update_switch_font(font_size)

    def adjust_font_size(self, width, height):
        base_font_size = 12
        scaling_factor = min(width/400, height/300)
        new_font_size = int(base_font_size * scaling_factor)
        self.apply_initial_styles(new_font_size)
        self.noise_suppression_component.adjust_component_sizes(width, height)

    def on_resize(self, event):
        if self.resize_after_id is not None:
            self.root.after_cancel(self.resize_after_id)
        self.resize_after_id = self.root.after(200, self.handle_resize, event.width, event.height)

    def handle_resize(self, width, height):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()    
        # print(width, height)
        self.adjust_font_size(width, height)

def main():
    root = ctk.CTk()
    ctk.set_appearance_mode("light")
    app = MainApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main()
