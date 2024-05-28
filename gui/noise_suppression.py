import tkinter as tk
from tkinter import ttk
import customtkinter as ctk

class NoiseSuppressionComponent:
    def __init__(self, root):
        self.root = root
        self.dry_set_slider_value = tk.DoubleVar()
        self.switch_var = ctk.StringVar(value="on")
        self.create_switch()
        self.create_slider()
        self.apply_styles()
    
    def create_switch(self):
        self.switch = ctk.CTkSwitch(self.root, text="Cancel Noise", command=self.switch_event,
                                    variable=self.switch_var, onvalue="on", offvalue="off")
        self.switch.place(relx=0.95, rely=0.05, anchor='ne')
    
    def switch_event(self):
        switch_state = self.switch_var.get()
        if switch_state == "on":
            self.slider.config(state=tk.NORMAL)
            print("Noise Cancellation is on")
        else:
            self.dry_set_slider_value.set(0)
            self.current_value_label.configure(text='0%')
            self.slider.config(state=tk.DISABLED)
            print("Noise Cancellation is off")

    def create_slider(self):
        slider_frame = tk.Frame(self.root)
        slider_frame.columnconfigure(0, weight=1)
        slider_frame.columnconfigure(1, weight=10)
        slider_frame.columnconfigure(2, weight=1)

        self.left_label = ttk.Label(slider_frame, text="0% (Dry)")
        self.left_label.grid(row=0, column=0, sticky=tk.W)

        self.slider = ttk.Scale(slider_frame, from_=0, to=100, orient="horizontal",
                                variable=self.dry_set_slider_value, command=self.slider_changed)
        self.slider.grid(row=0, column=1, sticky=tk.W + tk.E)

        self.right_label = ttk.Label(slider_frame, text="100% (Wet)")
        self.right_label.grid(row=0, column=2, sticky=tk.E)

        self.current_value_label = ttk.Label(slider_frame, text="Value: 0%")
        self.current_value_label.grid(column=1, row=1, sticky=tk.S)
        self.slider_changed(None)  # Initialize the label with the default value
        slider_frame.pack(fill='x', side="bottom")

    def slider_changed(self, event):
        new_value = self.round_to_nearest_multiple_of_step(int(self.get_current_value()[:-1]), 5)
        self.current_value_label.configure(text=str(new_value) + '%')  # change the 5 for different step
        self.dry_set_slider_value.set(new_value)

    def get_current_value(self):
        return '{:.0f}%'.format(self.dry_set_slider_value.get())

    def round_to_nearest_multiple_of_step(self, value, step):
        return round(value / step) * step

    def apply_styles(self):
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 12))
        style.configure("TButton", font=("Helvetica", 12))
        style.configure("TScale", font=("Helvetica", 12))
        self.left_label.configure(style="TLabel")
        self.right_label.configure(style="TLabel")
        self.current_value_label.configure(style="TLabel")
