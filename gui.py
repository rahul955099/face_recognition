import tkinter as tk
import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_face_recognition():
    subprocess.Popen([sys.executable, os.path.join(BASE_DIR, "detectface.py")])

def run_emotion_webcam():
    subprocess.Popen([sys.executable, os.path.join(BASE_DIR, "emotion_detection2.py")])

def run_emotion_image():
    subprocess.Popen([sys.executable, os.path.join(BASE_DIR, "emotion_detection3.py")])

def quit_app():
    root.destroy()

root = tk.Tk()
root.title("Face & Emotion Detection System")
root.geometry("400x300")

tk.Label(root, text="Select an Option", font=("Arial", 16)).pack(pady=20)

tk.Button(root, text="Face Recognition (Webcam)", width=30, command=run_face_recognition).pack(pady=5)
tk.Button(root, text="Emotion Detection (Webcam)", width=30, command=run_emotion_webcam).pack(pady=5)
tk.Button(root, text="Emotion Detection (Image)", width=30, command=run_emotion_image).pack(pady=5)
tk.Button(root, text="Quit", width=30, command=quit_app).pack(pady=20)

root.mainloop()
