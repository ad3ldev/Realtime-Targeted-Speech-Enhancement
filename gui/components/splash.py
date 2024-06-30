import tkinter as tk

class Splash(tk.Toplevel):
    def init(self, parent):
        super().init(parent)
        self.geometry("300x100")
        self.title("Loading...")
        label = tk.Label(self, text="Loading, please wait...", font=("Helvetica", 18))
        label.pack(expand=True)
        self.update()