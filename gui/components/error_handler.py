from tkinter import ttk
import config.objects_placement as op


class UserAlert:
    def __init__(self, root):
        self.alert_label = None
        self.root = root

    def alert_user(self, message):
        """
        Create a label to alert the user with a message.
        """
        self.alert_label = ttk.Label(self.root, text="                                                                 ",
                                     font=("Helvetica", 16), foreground="red")
        self.alert_label.place(relx=op.Alerts_Label_relx, rely=op.Alerts_Label_rely, anchor=op.Alerts_Label_anchor)

        self.alert_label = ttk.Label(self.root, text=message, font=("Helvetica", 16), foreground="red")
        self.alert_label.place(relx=op.Alerts_Label_relx, rely=op.Alerts_Label_rely, anchor=op.Alerts_Label_anchor)
