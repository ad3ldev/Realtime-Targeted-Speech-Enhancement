from tkinter import ttk
import tkinter as tk
import customtkinter as ctk
from components import InputDeviceComponent, NoiseSuppressionComponent, ReferenceVoiceComponent
import config.objects_placement as op
import config.color_modes as cm

color_mode = "light"

class MainApplication:
    resize_after_id = None

    def __init__(self, root):
        """
        Initialize the main application with the root window, set appearance mode, 
        initialize components, and create the submit button.
        """
        self.root = root

        self.root.title("Denoiser")
        self.root.geometry("400x250")
        self.root.minsize(325, 235)
        self.root.maxsize(600, 400)
    
        self.root.bind("<Configure>", self.on_resize)

        style = ttk.Style()
        style.theme_use('clam')  # Use a theme that supports color changes

        db_file = "reference_audios.db"
        user_id = "default"

        # Initialize components
        self.noise_suppression_component = NoiseSuppressionComponent(root, color_mode)
        self.reference_voice_component = ReferenceVoiceComponent(root, db_file, user_id, color_mode)
        self.input_device_component = InputDeviceComponent(root)

        self.set_mode(color_mode)

        self.get_components_values()
        self.apply_initial_styles()

        # Create the submit button
        self.submit_button = self.create_submit_button()


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
        submit_button = ttk.Button(self.root, text="Save", command=self.submit, style="Custom.TButton")
        submit_button.place(relx=op.submit_button_relx, rely=op.submit_button_rely, anchor=op.submit_button_anchor)
        return submit_button

    def apply_initial_styles(self, font_size=12):
        """
        Apply initial styles to the components with a specified font size.
        """
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", font_size))
        style.configure("TButton", font=("Helvetica", font_size))
        style.configure("TScale", font=("Helvetica", font_size))
        style.configure("TCombobox", font=("Helvetica", font_size))
        self.noise_suppression_component.update_switch_font(font_size)

    def adjust_font_size(self, width, height):
        """
        Adjust the font size based on the window's width and height.
        """
        base_font_size = 12
        scaling_factor = min(width / 350, height / 250)
        new_font_size = int(base_font_size * scaling_factor)
        self.apply_initial_styles(new_font_size)
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

    def submit(self):
        """
        Toggle the submit button state and disable or enable other components based on the current state.
        """
        if self.submit_button.cget("text") == "Save":
            values = self.get_components_values()
            if values.get("reference_audio")[0] == "default":
                print("Please select a username!!")
                return
            self.disable_components()
            self.submit_button.config(text="Edit")
            print(self.get_components_values())
        else:
            self.enable_components()
            self.submit_button.config(text="Save")

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
    ctk.set_appearance_mode(color_mode)
    root = ctk.CTk()
    app = MainApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main()


# import customtkinter as ctk

# # Set the appearance mode to "light"
# ctk.set_appearance_mode("light")  # Other options are "dark" and "system"

# # Create the main window
# root = ctk.CTk()

# # Set the window title
# root.title("Light Mode Window")

# # Set the window size
# root.geometry("400x300")

# # Create a label with some text
# label = ctk.CTkLabel(root, text="This is a light mode window!", font=("Arial", 20))
# label.pack(pady=20)

# # Create a button
# button = ctk.CTkButton(root, text="Click Me")
# button.pack(pady=10)

# # Run the application
# root.mainloop()
