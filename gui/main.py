import customtkinter as ctk
from noise_suppression import NoiseSuppressionComponent
from reference_voice import ReferenceVoiceComponent
from input_device import InputDeviceComponent


def main():
    root = ctk.CTk()
    ctk.set_appearance_mode("light")
    root.title("Denoiser")
    root.geometry("350x250")

    # Initialize components
    noise_suppression_component = NoiseSuppressionComponent(root)
    reference_voice_component = ReferenceVoiceComponent(root)
    input_device_component = InputDeviceComponent(root)

    root.mainloop()

if __name__ == "__main__":
    main()
