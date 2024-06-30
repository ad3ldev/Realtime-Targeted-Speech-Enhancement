from tkinter import ttk
import tkinter as tk
import customtkinter as ctk
import config.objects_placement as op
import config.color_modes as cm
from streaming.streamer import Streamer


def round_to_nearest_multiple_of_step(value, step):
    """
    Round the value to the nearest multiple of the step.
    """
    return round(value / step) * step


class NoiseSuppressionComponent:
    """
    Class to create a slider to set the noise suppression level and a switch to enable or disable noise suppression.
    """

    def __init__(self, root, bg_color, cfg, streamer: Streamer):
        self.cfg = cfg
        self.streamer = streamer
        self.slider = None
        self.left_label = None
        self.right_label = None
        self.slider_frame = None
        self.current_value_label = None
        self.root = root
        self.dry_wet_slider_value = tk.DoubleVar()
        self.switch_var = ctk.StringVar(value="off")
        # self.create_switch()
        self.create_slider()
        self.apply_styles()
        self.change_color_mode(bg_color)

        self.disable_components()
        self.enable_components()
        self.enabled_flag = True
        self.on_submit()

    def change_color_mode(self, mode="light"):
        """
        Change the color mode of the noise suppression component.
        """
        global bg_color, fg_color
        if mode == "light":
            bg_color = cm.bg_color_light
            fg_color = cm.fg_color_light
        elif mode == "dark":
            bg_color = cm.bg_color_dark
            fg_color = cm.fg_color_dark
        # self.switch._button_color = bg_color
        self.slider_frame.configure(background=bg_color)
        self.left_label.configure(background=bg_color, foreground=fg_color)
        self.right_label.configure(background=bg_color, foreground=fg_color)
        self.current_value_label.configure(background=bg_color, foreground=fg_color)

    def disable_components(self):
        """
        Disable the slider and the switch.
        """
        self.slider.config(state=tk.DISABLED)
        self.enabled_flag = False

    def enable_components(self):
        """
        Enable the slider and the switch.
        """
        self.slider.config(state=tk.NORMAL)
        self.enabled_flag = True

    def on_submit(self):
        """
        Return the value of the slider and the state of the switch.
        """
        return self.dry_wet_slider_value.get(), self.switch_var.get()

    def create_slider(self):
        """
        Create a slider to set the noise suppression level.
        """
        self.slider_frame = tk.Frame(self.root)
        self.slider_frame.columnconfigure(0, weight=1)
        self.slider_frame.columnconfigure(1, weight=10)
        self.slider_frame.columnconfigure(2, weight=1)

        self.left_label = ttk.Label(self.slider_frame, text="0% (Dry)")
        self.left_label.grid(row=0, column=0, sticky=tk.W)

        self.right_label = ttk.Label(self.slider_frame, text="100% (Wet)")
        self.right_label.grid(row=0, column=2, sticky=tk.E)

        self.slider = ttk.Scale(self.slider_frame, from_=0, to=100, orient="horizontal",
                                variable=self.dry_wet_slider_value, command=self.slider_changed)
        self.slider.grid(row=0, column=1, sticky=tk.W + tk.E)

        self.current_value_label = ttk.Label(self.slider_frame, text="Value: 0%")
        # self.current_value_label.configure(background=cm.bg_color_dark, foreground=cm.fg_color_dark)
        self.current_value_label.grid(column=1, row=1, sticky=tk.S)
        self.slider_changed(None)  # Initialize the label with the default value
        self.slider_frame.pack(fill='x', side="bottom", pady=op.NoiseSuppression_slider_pady)
        # self.slider_frame.place(relx=0.5, rely=0.2, anchor=tk.N)

    def slider_changed(self, event):
        """
        Event handler for the slider.
        """
        new_value = round_to_nearest_multiple_of_step(int(self.get_current_value()[:-1]), 1)
        self.current_value_label.configure(text=str(new_value) + '%')  # change the 5 for different step
        self.dry_wet_slider_value.set(new_value)
        new_value = new_value / 100
        print("Slider value: ", new_value)
        self.cfg["settings"]["dry"] = new_value
        self.streamer.dry = new_value

    def get_current_value(self):
        """
        Get the current value of the slider.
        """
        return '{:.0f}%'.format(self.dry_wet_slider_value.get())

    def apply_styles(self, font_size=12):
        """
        Apply styles to the noise suppression component.
        """
        style = ttk.Style()
        style.configure('TScale', font=("Helvetica", font_size))
        # style.configure('Horizontal.TScale')
        style.configure('TButton', font=("Helvetica", font_size))
        style.configure('TLabel', font=("Helvetica", font_size))
        style.configure('TFrame')
        # self.switch._label_font = ("Helvetica", font_size)

        self.left_label.configure(style="TLabel")
        self.right_label.configure(style="TLabel")
        self.current_value_label.configure(style="TLabel")

    def adjust_component_sizes(self, width, height):
        """
        Adjust the size of the components based on the window's width and height.
        """
        base_size = 14
        scaling_factor = min(width / 400, height / 300)
        new_size = int(base_size * scaling_factor)
        self.update_slider_size(new_size)

    def update_slider_size(self, size):
        """
        Update the size of the slider.
        """
        self.slider_frame.configure(height=size)
        self.slider.configure(length=size * 5)
