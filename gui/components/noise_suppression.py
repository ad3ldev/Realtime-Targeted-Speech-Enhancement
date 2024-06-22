from tkinter import ttk
import tkinter as tk
import customtkinter as ctk
import config.objects_placement as op

class NoiseSuppressionComponent:
    def __init__(self, root):
        self.root = root
        self.dry_set_slider_value = tk.DoubleVar()
        self.switch_var = ctk.StringVar(value="on")
        self.create_switch()
        self.create_slider()
        self.apply_styles()

    def create_switch(self):
        font = ("Helvetica", 20)
        self.switch = ctk.CTkSwitch(self.root, text="Cancel Noise", command=self.switch_event,
                                    variable=self.switch_var, onvalue="on", offvalue="off", font=font)
        self.switch.place(relx=op.NoiseSuppression_switch_relx, 
                          rely=op.NoiseSuppression_switch_rely, 
                          anchor=op.NoiseSuppression_switch_anchor)

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

    def update_switch_font(self, font_size):
        font = ("Helvetica", font_size)
        self.switch.configure(font=font)

    def update_switch_size(self, size):
        self.switch.configure(width=size, height=size)

    def create_slider(self):
        self.slider_frame = tk.Frame(self.root)
        self.slider_frame.columnconfigure(0, weight=1)
        self.slider_frame.columnconfigure(1, weight=10)
        self.slider_frame.columnconfigure(2, weight=1)

        self.left_label = ttk.Label(self.slider_frame, text="0% (Dry)")
        self.left_label.grid(row=0, column=0, sticky=tk.W)

        self.right_label = ttk.Label(self.slider_frame, text="100% (Wet)")
        self.right_label.grid(row=0, column=2, sticky=tk.E)

        self.slider = ttk.Scale(self.slider_frame, from_=0, to=100, orient="horizontal",
                                variable=self.dry_set_slider_value, command=self.slider_changed)
        self.slider.grid(row=0, column=1, sticky=tk.W + tk.E)

        self.current_value_label = ttk.Label(self.slider_frame, text="Value: 0%")
        self.current_value_label.grid(column=1, row=1, sticky=tk.S)
        self.slider_changed(None)  # Initialize the label with the default value
        self.slider_frame.pack(fill='x', side="bottom", pady=op.NoiseSuppression_slider_pady)
        # self.slider_frame.place(relx=0.5, rely=0.2, anchor=tk.N)

# def create_slider(self):
#     # Create the frame that will contain the slider and labels
#     self.slider_frame = tk.Frame(self.root)
    
#     # Calculate the positions and dimensions
#     frame_width = self.root.winfo_screenwidth()
#     label_width = frame_width / 12
#     slider_width = frame_width * 10 / 12

#     # Left label
#     self.left_label = ttk.Label(self.slider_frame, text="0% (Dry)")
#     self.left_label.place(relx=0, rely=0.5, anchor='w', width=label_width)

#     # Right label
#     self.right_label = ttk.Label(self.slider_frame, text="100% (Wet)")
#     self.right_label.place(relx=1, rely=0.5, anchor='e', width=label_width)

#     # Slider
#     self.slider = ttk.Scale(self.slider_frame, from_=0, to=100, orient="horizontal",
#                             variable=self.dry_set_slider_value, command=self.slider_changed)
#     self.slider.place(relx=0.5, rely=0.5, anchor='center', width=slider_width)

#     # Current value label (optional positioning, adjust as needed)
#     self.current_value_label = ttk.Label(self.slider_frame, text="Value: 0%")
#     self.current_value_label.place(relx=0.5, rely=1, anchor='n')
#     self.slider_changed(None)  # Initialize the label with the default value

#     # Place the frame in the middle of the window (adjust rely to move up/down)
#     self.slider_frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=1.0)


    def slider_changed(self, event):
        new_value = self.round_to_nearest_multiple_of_step(int(self.get_current_value()[:-1]), 5)
        self.current_value_label.configure(text=str(new_value) + '%')  # change the 5 for different step
        self.dry_set_slider_value.set(new_value)

    def get_current_value(self):
        return '{:.0f}%'.format(self.dry_set_slider_value.get())

    def round_to_nearest_multiple_of_step(self, value, step):
        return round(value / step) * step

    def apply_styles(self, font_size=12):
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", font_size))
        style.configure("TButton", font=("Helvetica", font_size))
        style.configure("TScale", font=("Helvetica", font_size))
        self.left_label.configure(style="TLabel")
        self.right_label.configure(style="TLabel")
        self.current_value_label.configure(style="TLabel")

    def adjust_component_sizes(self, width, height):
        base_size = 14
        scaling_factor = min(width/400, height/300)
        new_size = int(base_size * scaling_factor)
        self.update_switch_font(new_size)
        self.update_switch_size(new_size)
        self.update_slider_size(new_size)

    def update_slider_size(self, size):
        self.slider_frame.configure(height=size)
        self.slider.configure(length=size * 5)