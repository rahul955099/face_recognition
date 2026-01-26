import cv2
from deepface import DeepFace
from tkinter import filedialog, messagebox
import tkinter as tk

def main():
    root = tk.Tk()
    root.withdraw()   # hide tkinter main window

    # Select image
    img_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not img_path:
        return

    img = cv2.imread(img_path)
    if img is None:
        messagebox.showerror("Error", "Failed to load image")
        return

    # Analyze emotion
    result = DeepFace.analyze(
        img,
        actions=["emotion"],
        enforce_detection=False
    )

    if isinstance(result, list):
        result = result[0]

    emotion = result["dominant_emotion"].upper()

    # Show image first
    cv2.imshow("Selected Image", img)
    cv2.waitKey(1)  # allow OpenCV to render window

    # Show popup
    messagebox.showinfo(
        "Emotion Detected",
        f"Dominant Emotion: {emotion}"
    )

    # Close image window automatically after popup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
