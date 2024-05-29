from tkinter import ttk
import customtkinter as ctk
from sounddevice import query_devices, default

class InputDeviceComponent:
    padding_val = 10

    def __init__(self, root):
        self.root = root
        self.input_device_var = ctk.StringVar(value=self.get_default_input_device())
        self.create_input_device_dropdown()
        self.apply_styles()

    def get_input_devices(self):
        devices = query_devices()
        return [device["name"] for device in devices]

    def get_default_input_device(self):
        return default.device

    def create_input_device_dropdown(self):
        input_devices = self.get_input_devices()
        self.input_device_label = ttk.Label(self.root, text="Select Input Device")
        self.input_device_dropdown = ttk.Combobox(self.root, values=input_devices, textvariable=self.input_device_var, state="readonly")
        self.input_device_label.place(relx=0.05, rely=0.05, anchor='nw')
        self.input_device_dropdown.place(relx=0.05, rely=0.15, anchor='nw')

    def apply_styles(self):
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 12))
        style.configure("TCombobox", font=("Helvetica", 12))
        self.input_device_label.configure(style="TLabel")
        self.input_device_dropdown.configure(style="TCombobox")
