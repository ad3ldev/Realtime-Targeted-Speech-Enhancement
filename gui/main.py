from tkinter import ttk
import tkinter as tk
import customtkinter as ctk
from components import InputDeviceComponent, NoiseSuppressionComponent, ReferenceVoiceComponent
import config.objects_placement as op

class MainApplication:
    resize_after_id = None

    def __init__(self, root):
        # Configurations:
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

        self.get_components_values()
        self.apply_initial_styles()

        # Create the submit button
        self.submit_button = self.create_submit_button()
        # self.save_user_button = ttk.Button(self.root, text="Save", command=self.submit, style="Custom.TButton")
        # self.save_user_button.place(relx=op.submit_button_relx,
        #                             rely=op.submit_button_rely,
        #                             anchor=op.submit_button_anchor)
        
    # Create a button to return the user data: username, reference audio, and noise suppression settings, then toggles the button state and disables all the other buttons, slider and comboboxes until the button is pressed again to stop the application:
    def create_submit_button(self):
        submit_button = ttk.Button(self.root, text="Save", command=self.submit, style="Custom.TButton")
        submit_button.place(relx=op.submit_button_relx,
                            rely=op.submit_button_rely,
                            anchor=op.submit_button_anchor)
        return submit_button


    def apply_initial_styles(self, font_size=12):
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", font_size))
        style.configure("TButton", font=("Helvetica", font_size))
        style.configure("TScale", font=("Helvetica", font_size))
        style.configure("TCombobox", font=("Helvetica", font_size))
        self.noise_suppression_component.update_switch_font(font_size)

    def adjust_font_size(self, width, height):
        base_font_size = 12
        scaling_factor = min(width/350, height/250)
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

    def submit(self):
        if self.submit_button.cget("text") == "Save":
            values = self.get_components_values()
            if values.get("reference_audio")[0] == "default":
                print("Please select a username!!")
                return
            self.disable_components()
            self.submit_button.config(text="Edit")
            # print then return components values:
            print(self.get_components_values())
            return self.get_components_values()
        else:
            self.enable_components()
            self.submit_button.config(text="Save")
    
    def get_components_values(self):
        return {
            "input_device": self.input_device_component.on_submit(),
            "reference_audio": self.reference_voice_component.on_submit(),
            "noise_suppression": self.noise_suppression_component.on_submit()
        }

    def disable_components(self):
        self.noise_suppression_component.disable_components()
        self.reference_voice_component.disable_components()
        self.input_device_component.disable_components()

    def enable_components(self):
        self.noise_suppression_component.enable_components()
        self.reference_voice_component.enable_components()
        self.input_device_component.enable_components()

def main():
    root = ctk.CTk()
    ctk.set_appearance_mode("light")
    app = MainApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main()
