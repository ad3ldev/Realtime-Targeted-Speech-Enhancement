import shutil
import tkinter as tk
from tkinter import ttk, filedialog
import os
import config.objects_placement as op
import config.color_modes as cm
from dao.database import create_connection, create_table, select_all_users, save_user, get_path_from_file_name, \
    get_file_name_by_user_id, update_reference, get_file_path_by_user_id
from tkinter import font
from gui.components.error_handler import UserAlert


class ReferenceVoiceComponent:
    """
    Class to create a dropdown menu to select the reference voice.
    """

    def __init__(self, root, db_file, user_id, bg_color):
        """
        Initialize the Reference Voice Component with the root window, database file, and user id.
        """
        self.ok_button = None
        self.new_reference_name_label = None
        self.upload_button = None
        self.submit_frame = None
        self.upload_frame = None
        self.new_user_frame = None
        self.new_username_entry = None
        self.last_uploaded_audio = None
        self.newuser_window = None
        self.upload_reference_button = None
        self.change_reference_label = None
        self.reference_voice_name = None
        self.reference_voice_label = None
        self.save_user_button = None
        self.new_username_label = None
        self.saved_users_dropdown = None
        self.saved_users_var = None
        self.username_label = None
        self.new_file_path = None
        self.root = root
        self.reference_audio_path = tk.StringVar()
        self.reference_voice_name_var = tk.StringVar(value="No one selected!")
        self.new_username_var = tk.StringVar(value="")
        self.bold_font = font.Font(family="Helvetica", size=12, weight="bold")
        self.bg_color = bg_color

        self.new_reference_name_var = tk.StringVar(value="No file selected!")
        self.new_reference_audio_path = tk.StringVar()
        self.db_file = db_file
        self.user_id = user_id
        self.selected_reference = None
        self.audio_storage_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'audio_files')
        self.init_database()
        self.create_users_section()
        self.create_audio_storage_dir()

        self.on_reference_selected()
        self.disable_components()
        self.enable_components()
        self.on_submit()

        self.apply_styles()

    def disable_components(self):
        """
        Disable the components of the Reference Voice Component.
        """
        self.saved_users_dropdown.configure(state="disabled")
        self.upload_reference_button.configure(state="disabled")
        self.save_user_button.configure(state="disabled")

    def enable_components(self):
        """
        Enable the components of the Reference Voice Component.
        """
        self.saved_users_dropdown.configure(state="normal")
        self.upload_reference_button.configure(state="normal")
        self.save_user_button.configure(state="normal")

    def on_submit(self):
        """
        Return the user id and the reference audio path.
        """
        self.on_reference_selected()
        return self.user_id, self.reference_audio_path.get()

    def create_audio_storage_dir(self):
        """
        Create the audio storage directory if it does not exist.
        """
        if not os.path.exists(self.audio_storage_dir):
            os.makedirs(self.audio_storage_dir)

    def get_path_from_file_name(self, file_name):
        """
        Get the path of the audio file from the file name.
        """
        conn = create_connection(self.db_file)
        with conn:
            path = get_path_from_file_name(conn, file_name)
        return path

    def on_reference_selected(self):
        '''
        Get the reference audio path when a reference is selected from the dropdown.
        '''
        self.selected_reference = self.reference_voice_name_var.get()
        # print(f"Selected reference: {self.selected_reference}")
        path = self.get_path_from_file_name(self.selected_reference)
        self.reference_audio_path.set(path)

    def on_user_selected(self, event):
        """
        Get the user id when a user is selected from the dropdown.
        """
        self.user_id = self.saved_users_var.get()
        self.reference_voice_name_var.set(get_file_name_by_user_id(create_connection(self.db_file), self.user_id))
        UserAlert.alert_user(self, "")
        print(f"Selected user: {self.user_id}")

    def create_users_section(self):
        """
        Initialize and place all widgets related to user selection and reference voice configuration in the main window.
        """
        # Username Section
        # Label
        self.username_label = ttk.Label(self.root, text="Select User")
        self.username_label.place(relx=op.ReferenceVoice_SelectUser_Label_relx,
                                  rely=op.ReferenceVoice_SelectUser_Label_rely,
                                  anchor=op.ReferenceVoice_SelectUser_Label_anchor)
        # Dropdown
        self.saved_users_var = tk.StringVar()
        self.saved_users_dropdown = ttk.Combobox(self.root, textvariable=self.saved_users_var, state="readonly")
        self.saved_users_dropdown.place(relx=op.ReferenceVoice_SelectUser_Dropdown_relx,
                                        rely=op.ReferenceVoice_SelectUser_Dropdown_rely,
                                        anchor=op.ReferenceVoice_SelectUser_Dropdown_anchor)
        self.saved_users_dropdown.bind("<<ComboboxSelected>>", self.on_user_selected)
        self.load_saved_users()

        # New Username Section
        # Label
        self.new_username_label = ttk.Label(self.root, text="Not listed?")
        self.new_username_label.place(relx=op.ReferenceVoice_NewUser_Label_relx,
                                      rely=op.ReferenceVoice_NewUser_Label_rely,
                                      anchor=op.ReferenceVoice_NewUser_Label_anchor)
        # Button
        if self.bg_color == "dark":
            self.save_user_button = ttk.Button(self.root, text="Create new Account",
                                               command=lambda: self.create_new_account(cm.bg_color_dark),
                                               style="Custom.TButton")
        elif self.bg_color == "light":
            self.save_user_button = ttk.Button(self.root, text="Create new Account",
                                               command=lambda: self.create_new_account(cm.bg_color_light),
                                               style="Custom.TButton")
        self.save_user_button.place(relx=op.ReferenceVoice_SaveUser_Button_relx,
                                    rely=op.ReferenceVoice_SaveUser_Button_rely,
                                    anchor=op.ReferenceVoice_SaveUser_Button_anchor)
        # Reference Voice Section
        # Label
        self.reference_voice_label = ttk.Label(self.root, text="Reference Voice")
        self.reference_voice_label.place(relx=op.ReferenceVoice_SelectReference_Label_relx,
                                         rely=op.ReferenceVoice_SelectReference_Label_rely,
                                         anchor=op.ReferenceVoice_SelectReference_Label_anchor)
        # Name
        self.reference_voice_name = ttk.Label(self.root, textvariable=self.reference_voice_name_var,
                                              font=self.bold_font)
        self.reference_voice_name.place(relx=op.ReferenceVoice_SelectReference_Label_relx,
                                        rely=op.ReferenceVoice_SelectReference_Label_rely + 0.1,
                                        anchor=op.ReferenceVoice_SelectReference_Label_anchor)
        # Change Reference Section
        # Label
        self.change_reference_label = ttk.Label(self.root, text="Change Reference?")
        self.change_reference_label.place(relx=op.ReferenceVoice_ChangeReference_Label_relx,
                                          rely=op.ReferenceVoice_ChangeReference_Label_rely,
                                          anchor=op.ReferenceVoice_ChangeReference_Label_anchor)
        # Button
        self.upload_reference_button = ttk.Button(self.root, text="Upload Audio", command=self.update_audio,
                                                  style="Custom.TButton")
        self.upload_reference_button.place(relx=op.ReferenceVoice_ChangeReference_Label_relx,
                                           rely=op.ReferenceVoice_ChangeReference_Label_rely + 0.1,
                                           anchor=op.ReferenceVoice_ChangeReference_Label_anchor)

        # self.load_saved_references()

    def create_new_account(self, bg_color=cm.bg_color_dark):  # CHANGE THIS INDEPENDENTLY FOR NEW WINDOW MODE.
        """
        Create a new account for a new user: username - audio file.
        """
        self.newuser_window = tk.Toplevel(self.root)
        self.newuser_window.title("New User")
        self.newuser_window.geometry("300x130")  # Adjut the secondary window size
        self.newuser_window.resizable(False, False)
        self.newuser_window.configure(bg=bg_color)

        # self.newuser_window.configure(bg=bg_color)
        self.last_uploaded_audio = None

        # Username Field
        self.new_username_label = ttk.Label(self.newuser_window, text="Enter Username")
        self.new_username_label.pack()
        self.new_username_entry = ttk.Entry(self.newuser_window, textvariable=self.new_username_var)
        self.new_username_entry.pack(fill='x', padx=10, pady=5)

        self.new_user_frame = tk.Frame(self.newuser_window, bg=bg_color)

        self.upload_frame = tk.Frame(self.new_user_frame, bg=bg_color)
        self.upload_frame.columnconfigure(0, weight=4)
        self.upload_frame.columnconfigure(1, weight=8)

        self.submit_frame = tk.Frame(self.new_user_frame, bg=bg_color)
        self.submit_frame.columnconfigure(0, weight=4)

        # Upload Audio Button
        self.upload_button = ttk.Button(self.upload_frame, text="Upload Audio", command=self.upload_audio)
        self.upload_button.grid(row=0, column=0)
        self.new_reference_name_label = ttk.Label(self.upload_frame, textvariable=self.new_reference_name_var)
        self.new_reference_name_label.grid(row=0, column=1)
        self.upload_frame.pack(fill='x', pady=5)

        # Ok Button
        self.ok_button = tk.Button(self.submit_frame, text="   Ok   ", command=self.submit_new_user)
        self.ok_button.grid(row=0, column=0)
        self.submit_frame.pack()

        self.new_user_frame.pack(side='bottom')

    def upload_audio(self):
        '''
        Upload the reference audio of the new user.
        '''
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.m4a")])
        if file_path:
            file_name = os.path.basename(file_path)
            self.new_file_path = os.path.join(self.audio_storage_dir, file_name)

            shutil.copy(file_path, self.new_file_path)
            self.new_reference_audio_path.set(self.new_file_path)
            print(f"Selected audio file: {file_path}")
            self.new_reference_name_var.set(os.path.basename(self.new_file_path))
            self.last_uploaded_audio = self.new_file_path

    def update_audio(self):
        """
        Update the reference audio of the selected user.
        """
        if self.user_id == "default":
            # Alert User instead of printing:
            UserAlert.alert_user(self, "Please select a user first!")
            return
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.m4a")])
        if file_path:
            file_to_remove = get_file_path_by_user_id(create_connection(self.db_file), self.user_id)

            file_name = os.path.basename(file_path)
            new_name = self.user_id + os.path.splitext(file_name)[1]
            self.new_file_path = os.path.join(self.audio_storage_dir, self.user_id + os.path.splitext(file_name)[1])

            shutil.copy(file_path, self.new_file_path)
            self.reference_audio_path.set(self.new_file_path)
            print(f"Selected audio file: {file_path}")
            self.new_reference_name_var.set(os.path.basename(self.new_file_path))
            self.last_uploaded_audio = self.new_file_path

            os.remove(file_to_remove)
            update_reference(create_connection(self.db_file), self.user_id, new_name, self.new_file_path)
            self.reference_voice_name_var.set(new_name)

    def submit_new_user(self):
        """
        Submit the new user details to the database.
        """
        # Get the username
        username = self.new_username_var.get()
        print(f"Username: {username}")
        print(f"File Path: {self.reference_audio_path.get()}")

        # Check if all fields are filled
        if not username:
            UserAlert.alert_user(self, "Username cannot be empty!")
            return
        if self.new_reference_audio_path.get() == "" or self.new_reference_audio_path.get() is None:
            UserAlert.alert_user(self, "Audio file not uploaded!")
            return

        UserAlert.alert_user(self, "")

        self.reference_audio_path.set(self.new_reference_audio_path.get())
        self.new_reference_audio_path.set("")
        # Get file name & absolute path
        file_name = os.path.basename(self.reference_audio_path.get())
        new_name = username + os.path.splitext(file_name)[1]
        directory = os.path.dirname(self.reference_audio_path.get())
        file_path = os.path.join(directory, new_name)

        print(f"Username: {username}, File Name: {new_name}, File Path: {file_path}")
        conn = create_connection(self.db_file)
        with conn:
            try:
                save_user(conn, username, new_name, file_path)
            except:
                UserAlert.alert_user(self, "User already Exists!!")
                print(self.last_uploaded_audio)
                if self.last_uploaded_audio is not None and os.path.exists(self.last_uploaded_audio):
                    os.remove(self.last_uploaded_audio)
                    self.last_uploaded_audio = None
                    print("Deleted the last uploaded audio file")

        try:
            os.rename(self.reference_audio_path.get(), file_path)
        except FileNotFoundError:
            print("Nothing to rename!")
            return
        self.load_saved_users()
        self.newuser_window.destroy()
        self.reference_audio_path = tk.StringVar()
        self.new_username_var = tk.StringVar(value="")

    def load_saved_users(self):
        """
        Load all saved users from the database.
        """
        conn = create_connection(self.db_file)
        with conn:
            users = select_all_users(conn)
        user_names = [user[0] for user in users]
        self.saved_users_dropdown['values'] = user_names

    def apply_styles(self, bg_color=cm.bg_color_light, fg_color=cm.fg_color_light):
        """
        Apply styles to the components of the Reference Voice Component.
        """
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12), background=bg_color, foreground=fg_color)
        style.configure("TLabel", font=("Helvetica", 12), background=bg_color, foreground=fg_color)
        style.configure("TCombobox", font=("Helvetica", 12))
        style.configure("Custom.TButton", font=("Helvetica", 12), background=bg_color, foreground=fg_color)

        self.username_label.configure(style="TLabel")
        self.saved_users_dropdown.configure(style="TCombobox")
        self.new_username_label.configure(style="TLabel")
        self.save_user_button.configure(style="Custom.TButton")
        self.reference_voice_label.configure(style="TLabel")
        self.reference_voice_name.configure(style="TLabel")
        self.change_reference_label.configure(style="TLabel")
        self.upload_reference_button.configure(style="Custom.TButton")
        self.new_username_label.configure(style="TLabel")

    def change_color_mode(self, mode):
        """
        Change the color mode of the Reference Voice Component.
        """
        if mode == "dark":
            self.apply_styles(cm.bg_color_dark, cm.fg_color_dark)
        elif mode == "light":
            self.apply_styles(cm.bg_color_light, cm.fg_color_light)

    def init_database(self):
        conn = create_connection(self.db_file)
        with conn:
            create_table(conn)
