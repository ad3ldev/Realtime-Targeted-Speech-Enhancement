from tkinter import ttk
import customtkinter as ctk
from components import InputDeviceComponent, NoiseSuppressionComponent, ReferenceVoiceComponent, Splash
import config.objects_placement as op
import config.color_modes as cm
from components.error_handler import UserAlert
import json
import threading
import sys
from streaming.streaming import initialize_streamer, start_streaming, stop_streaming

color_mode = "light"


def apply_initial_styles(font_size=12):
    """
    Apply initial styles to the components with a specified font size.
    """
    style = ttk.Style()
    style.configure("TLabel", font=("Helvetica", font_size))
    style.configure("TButton", font=("Helvetica", font_size))
    style.configure("TScale", font=("Helvetica", font_size))
    style.configure("TCombobox", font=("Helvetica", font_size))


class MainApplication:
    resize_after_id = None

    def __init__(self, root, cfg):
        """
        Initialize the main application with the root window, set appearance mode,
        initialize components, and create the submit button.
        """
        self.cfg = cfg
        
        self.streamer = initialize_streamer(cfg)
        self.stop_streaming_event = threading.Event()
        self.stop_streaming_event.clear()
        self.release_resource_event = threading.Event()
        self.release_resource_event.set()
        # self.streaming_thread = start_streaming(self.streamer, cfg, self.stop_streaming_event)
                
        self.switch = None
        self.root = root

        self.root.title("Denoiser")
        self.root.geometry("400x250")
        self.root.minsize(325, 235)
        self.root.maxsize(600, 400)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.switch_var = ctk.StringVar(value="off")
        self.root.bind("<Configure>", self.on_resize)

        style = ttk.Style()
        style.theme_use('clam')  # Use a theme that supports color changes

        db_file = "reference_audios.db"
        user_id = "default"
        
        # Initialize components
        self.noise_suppression_component = NoiseSuppressionComponent(root, color_mode, cfg, self.streamer)
        self.reference_voice_component = ReferenceVoiceComponent(root, db_file, user_id, color_mode, cfg, self.streamer)
        self.input_device_component = InputDeviceComponent(root, cfg, self.streamer, self.stop_streaming_event, self.release_resource_event)

        self.set_mode(color_mode)

        self.get_components_values()
        apply_initial_styles()

        # Create the submit button
        self.submit_button = self.create_submit_button()
        
        
    def on_closing(self):
        stop_streaming(self.stop_streaming_event, self.release_resource_event)
        print("App Closed!")
        self.root.destroy()
    
    def set_mode(self, mode):
        self.noise_suppression_component.change_color_mode(mode)
        self.reference_voice_component.change_color_mode(mode)

        if mode == "light":
            ctk.set_appearance_mode(cm.bg_color_light)
        elif mode == "dark":
            ctk.set_appearance_mode(cm.bg_color_dark)

    def create_submit_button(self):
        """
        Create a button that toggles its state and text when clicked,
        disabling all other components until clicked again.
        """
        font = ("Helvetica", 20)
        self.switch = ctk.CTkSwitch(self.root, text="Cancel Noise", command=self.switch_event,
                                    variable=self.switch_var, onvalue="on", offvalue="off", font=font)
        self.switch.place(relx=op.NoiseSuppression_switch_relx, rely=op.NoiseSuppression_switch_rely,
                          anchor=op.NoiseSuppression_switch_anchor)
        #
        return self.switch

    """
    Switch event replaces the Submit Functionality
    Need to make sure that fields are set.
    """

    def switch_event(self):
        self.streamer.enhance = True if self.switch_var.get() == "on" else False
        print("streamer.enhance: ", self.streamer.enhance)

    def adjust_font_size(self, width, height):
        """
        Adjust the font size based on the window's width and height.
        """
        base_font_size = 12
        scaling_factor = min(width / 350, height / 250)
        new_font_size = int(base_font_size * scaling_factor)
        apply_initial_styles(new_font_size)
        self.noise_suppression_component.adjust_component_sizes(width, height)

    def on_resize(self, event):
        """
        Handle the resize event by scheduling a resize adjustment after a delay.
        """
        if self.resize_after_id is not None:
            self.root.after_cancel(self.resize_after_id)
        self.resize_after_id = self.root.after(200, self.handle_resize, event.width, event.height)

    def handle_resize(self, width, height):
        """
        Adjust the component sizes after resizing the window.
        """
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        self.adjust_font_size(width, height)

    def get_components_values(self):
        """
        Get the current values from all components.
        """
        return {
            "input_device": self.input_device_component.on_submit(),
            "reference_audio": self.reference_voice_component.on_submit(),
            "noise_suppression": self.noise_suppression_component.on_submit()
        }

    def disable_components(self):
        """
        Disable all components.
        """
        self.noise_suppression_component.disable_components()
        self.reference_voice_component.disable_components()
        self.input_device_component.disable_components()

    def enable_components(self):
        """
        Enable all components.
        """
        self.noise_suppression_component.enable_components()
        self.reference_voice_component.enable_components()
        self.input_device_component.enable_components()

def main():
    """
    The main function to initialize and run the application.
    """
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)
    # streamer = initialize_streamer(cfg, "D:/Graduation Project/muhammad-embedding-cropped.wav")
    
    # stop_event = threading.Event()
    # stop_event.clear()
    
    # streaming_thread = start_streaming(streamer, cfg, stop_event)
    
    ctk.set_appearance_mode(color_mode)
    root = ctk.CTk()
    app = MainApplication(root, cfg)
    root.mainloop()
    
    # streaming_thread.join()


if __name__ == "__main__":
    main()