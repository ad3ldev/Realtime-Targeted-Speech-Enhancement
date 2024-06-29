import os
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install_dependencies():
    # Install packages from requirements.txt
    if os.path.exists('requirements.txt'):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    # Install tkinter separately if not present
    try:
        import tkinter
    except ImportError:
        if sys.platform == "darwin":
            print("Installing tkinter via brew...")
            subprocess.check_call(["brew", "install", "python-tk"])
        else:
            print("tkinter is not installed and cannot be installed automatically on this platform. Please install it manually.")

    # Additional handling for PyAudio on macOS
    if sys.platform == "darwin":
        print("Installing portaudio via brew for PyAudio...")
        subprocess.check_call(["brew", "install", "portaudio"])
        install_package("pyaudio")

    # The following packages are part of the standard library and do not need to be installed:
    # shutil
    # sqlite3

if __name__ == "__main__":
    install_dependencies()
