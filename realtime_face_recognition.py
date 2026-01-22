import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

# ---------------- CONFIG ----------------
IMG_SIZE = 224
MODEL_PATH = "face_recognition_model.h5"
CLASS_NAMES_PATH = "class_names.txt"
THRESHOLD = 0.75   # confidence threshold: adjust 0.65–0.85

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ---------------- FACE DETECTOR ----------------
detector = MTCNN()
cap = cv2.VideoCapture(0)

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)

        # Crop face & preprocess
        face_img = rgb[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        # Predict
        preds = model.predict(face_img, verbose=0)
        conf = np.max(preds)
        class_id = np.argmax(preds)

        if conf > THRESHOLD:
            name = class_names[class_id]
            label = f"{name} ({conf*100:.1f}%)"
            color = (0, 255, 0)  # green
        else:
            label = "Unknown"
            color = (0, 0, 255)  # red

        # Draw rectangle & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Deep Learning Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
