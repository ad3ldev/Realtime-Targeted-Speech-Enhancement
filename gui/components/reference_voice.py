import shutil
import tkinter as tk
from tkinter import ttk, filedialog
import os
from dao.database import create_connection, create_table, insert_reference_audio, select_all_reference_audios, select_all_users, insert_user, get_path_from_file_name

class ReferenceVoiceComponent:
    padding_val = 5

    def __init__(self, root, db_file, user_id):
        self.root = root
        self.reference_audio_path = tk.StringVar()
        self.db_file = db_file
        self.user_id = user_id
        self.selected_reference = None
        self.audio_storage_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'audio_files')
        self.init_database()
        # self.create_saved_references_dropdown()
        self.create_users_section()
        self.create_references_section()
        self.apply_styles()
        self.create_audio_storage_dir()


    def create_audio_storage_dir(self):
        if not os.path.exists(self.audio_storage_dir):
            os.makedirs(self.audio_storage_dir)

    def create_upload_button(self, rel_x, rel_y):
        self.upload_button = ttk.Button(self.root, text="Upload Reference Audio", command=self.upload_audio, style="Custom.TButton", padding=int(self.padding_val))
        self.upload_button.place(relx=rel_x, rely=rel_y, anchor=tk.W)

    # def create_reference_audio_label(self):
    #     self.reference_audio_label = ttk.Label(self.root, text="Nothing selected", style="Custom.TButton", padding=int(self.padding_val))
    #     self.reference_audio_label.place(relx=0.55, rely=0.5, anchor=tk.W)

    def get_path_from_file_name(self, file_name):
        conn = create_connection(self.db_file)
        with conn:
            path = get_path_from_file_name(conn, file_name)
        return path

    def on_reference_selected(self, event):
        self.selected_reference = self.saved_references_var.get()
        print(f"Selected reference: {self.selected_reference}")
        path = self.get_path_from_file_name(self.selected_reference)
        self.reference_audio_path.set(path)
        print(f"Its path: {self.reference_audio_path.get()}")

    def create_references_section(self):
        self.reference_audio_label = ttk.Label(self.root, text="Select Your Reference")
        self.reference_audio_label.place(relx=0.05, rely=0.5, anchor=tk.N + tk.W)

        self.saved_references_var = tk.StringVar()
        self.saved_references_dropdown = ttk.Combobox(self.root, textvariable=self.saved_references_var, state="readonly")
        self.saved_references_dropdown.place(relx=0.05, rely=0.6, anchor=tk.N + tk.W)
        self.saved_references_dropdown.bind("<<ComboboxSelected>>", self.on_reference_selected)
        self.load_saved_references()

        self.new_reference_label = ttk.Label(self.root, text="Not listed? Upload one!")
        self.new_reference_label.place(relx=0.45, rely=0.5, anchor=tk.N + tk.W)

        self.create_upload_button(0.45, 0.65)

    
    def on_user_selected(self, event):
        self.user_id = self.saved_users_var.get()
        print(f"Selected user: {self.user_id}")


    def create_users_section(self):
        self.username_label = ttk.Label(self.root, text="Select User")
        self.username_label.place(relx=0.05, rely=0.25, anchor=tk.N + tk.W)

        self.saved_users_var = tk.StringVar()
        self.saved_users_dropdown = ttk.Combobox(self.root, textvariable=self.saved_users_var, state="readonly")
        self.saved_users_dropdown.place(relx=0.05, rely=0.35, anchor=tk.N + tk.W)
        self.saved_users_dropdown.bind("<<ComboboxSelected>>", self.on_user_selected)
        self.load_saved_users()

        self.new_username_label = ttk.Label(self.root, text="Not listed? Enter new username!")
        self.new_username_label.place(relx=0.45, rely=0.25, anchor=tk.N + tk.W)

        self.root.update_idletasks()
        window_width = self.root.winfo_width()
        print("window_width:", window_width)

        self.new_username_entry = ttk.Entry(self.root)
        self.new_username_entry.place(relx=0.45, rely=0.35, anchor=tk.N + tk.W)
        self.new_username_entry.config(width=int(window_width * 0.1))

        self.save_user_button = ttk.Button(self.root, text="Save", command=self.save_user, style="Custom.TButton")
        self.save_user_button.place(relx=0.75, rely=0.35, anchor=tk.N + tk.W)
        self.save_user_button.config(width=int(window_width * 0.04))

    def save_user(self):
        new_user = self.new_username_entry.get()
        if new_user:
            conn = create_connection(self.db_file)
            with conn:
                insert_user(conn, new_user)
            self.load_saved_users()
            self.new_username_entry.delete(0, tk.END)

    # def upload_audio(self):
    #     file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.m4a")])
    #     if file_path:
    #         self.reference_audio_path.set(file_path)
    #         print(f"Selected file: {self.reference_audio_path.get()}")
    #         file_name = os.path.basename(file_path)
    #         # self.reference_audio_label.configure(text=file_name)
    #         self.save_reference_audio(file_name, file_path)


    def upload_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.m4a")])
        if file_path:
            file_name = os.path.basename(file_path)
            new_file_path = os.path.join(self.audio_storage_dir, file_name)
            
            shutil.copy(file_path, new_file_path)
            self.reference_audio_path.set(new_file_path)
            # print(f"Selected file: {self.reference_audio_path.get()}")
            self.save_reference_audio(file_name, new_file_path)


    def save_reference_audio(self, file_name, file_path):
        conn = create_connection(self.db_file)
        with conn:
            insert_reference_audio(conn, self.user_id, file_name, file_path)
        self.load_saved_references()


    # def load_saved_references(self):
    #     conn = create_connection(self.db_file)
    #     with conn:
    #         references = select_all_reference_audios(conn, self.user_id)
    #     file_names = [ref[0] for ref in references]  # file_name is the 1st column
    #     self.saved_references_dropdown['values'] = file_names

    def load_saved_references(self):
        if not os.path.exists(self.audio_storage_dir):
            os.makedirs(self.audio_storage_dir)
        # List all audio files in the directory
        audio_files = [f for f in os.listdir(self.audio_storage_dir) if os.path.isfile(os.path.join(self.audio_storage_dir, f))]
        self.saved_references_dropdown['values'] = audio_files

    
    def load_saved_users(self):
        conn = create_connection(self.db_file)
        with conn:
            users = select_all_users(conn)
        user_names = [user[0] for user in users]
        self.saved_users_dropdown['values'] = user_names


    def apply_styles(self):
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12))
        style.configure("TLabel", font=("Helvetica", 12))
        # style.configure("TEntry", font=("Helvetica", 12))
        self.upload_button.configure(style="TButton")
        self.reference_audio_label.configure(style="TLabel")
        self.saved_references_dropdown.configure(style="TCombobox")
        # self.username_entry.configure(style="TEntry")
        # self.username_label.configure(style="TLabel")

    def init_database(self):
        conn = create_connection(self.db_file)
        with conn:
            create_table(conn)
    
