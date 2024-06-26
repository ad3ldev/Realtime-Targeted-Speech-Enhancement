from tkinter import ttk
import customtkinter as ctk
from sounddevice import query_devices, default
import config.objects_placement as op
import config.color_modes as cm
import pyaudio


class InputDeviceComponent:
    '''
    Class to create a dropdown menu to select the input device.
    '''
    def __init__(self, root):
        '''
        Initialize the input device component with the root window.
        '''
        self.root = root
        self.input_device_var = ctk.StringVar(value=self.get_default_input_device())
        self.create_input_device_dropdown()
        self.apply_styles()

        self.disable_components()
        self.enable_components()
        self.on_submit()


    def disable_components(self):
        '''
        Disable the input device dropdown.
        '''
        self.input_device_dropdown.configure(state="disabled")


    def enable_components(self):
        '''
        Enable the input device dropdown.
        '''
        self.input_device_dropdown.configure(state="readonly")


    def on_submit(self):
        '''
        Return the selected input device.
        '''
        return self.input_device_var.get()
    

    def get_input_devices(self):
        '''
        Get the list of input devices.
        '''
        p = pyaudio.PyAudio()
        input_devices = []
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            # if(device_info['maxInputChannels'] > 0 and device_info['hostApi'] == 0):
            if(0 < device_info['maxInputChannels'] < 3):
                input_devices.append(device_info['name'])
        return input_devices


    def get_default_input_device(self):
        '''
        Get the default input device.
        '''
        default_device_index = default.device[0]  # Default input device index
        devices = query_devices()
        if default_device_index < len(devices):
            return devices[default_device_index]['name']
        return ""

    
    def create_input_device_dropdown(self):
        '''
        Create the input device dropdown.
        '''
        input_devices = self.get_input_devices()
        self.input_device_label = ttk.Label(self.root, text="Select Input Device")
        self.input_device_dropdown = ttk.Combobox(self.root, values=input_devices, textvariable=self.input_device_var, state="readonly")
        self.input_device_label.place(relx=op.InputDevice_Label_relx, 
                                      rely=op.InputDevice_Label_rely, 
                                      anchor=op.InputDevice_Label_anchor)
        self.input_device_dropdown.place(relx=op.InputDevice_Dropdown_relx,
                                         rely=op.InputDevice_Dropdown_rely,
                                         anchor=op.InputDevice_Dropdown_anchor)
        # Bind the resizing event
        self.root.bind("<Configure>", self.adjust_combobox_width)


    def adjust_combobox_width(self, event):
        '''
        Adjust the width of the input device dropdown.
        '''
        new_width = int(self.root.winfo_width() * 0.14) # CHANGE THIS VALUE for big dropdwn menu width
        self.input_device_dropdown.config(width=new_width)


    def apply_styles(self):
        '''
        Apply styles to the input device component.
        '''
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 12))
        style.configure("TCombobox", font=("Helvetica", 12))
        self.input_device_label.configure(style="TLabel")
        self.input_device_dropdown.configure(style="TCombobox")
