import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

IMG_SIZE = 224
MODEL_PATH = "face_recognition_model.h5"
CLASS_NAMES_PATH = "class_names.txt"
THRESHOLD = 0.75

# Load model & class names
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

detector = MTCNN()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, int(frame.shape[0]*640/frame.shape[1])))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        faces = detector.detect_faces(rgb)
    except:
        continue

    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        if w < 48 or h < 48:
            continue

        face_img = rgb[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        preds = model.predict(face_img, verbose=0)
        conf = np.max(preds)
        class_id = np.argmax(preds)

        if conf > THRESHOLD:
            label = f"{class_names[class_id]} ({conf*100:.1f}%)"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
