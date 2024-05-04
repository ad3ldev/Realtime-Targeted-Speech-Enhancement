import os

get_file_name_without_extension = lambda file_path: os.path.splitext(os.path.basename(file_path))[0]