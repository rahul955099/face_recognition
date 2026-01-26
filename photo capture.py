import cv2
import os
import time
import json
import numpy as np

import pickle

# ===== USER INPUT =====
name = input("Enter Name: ").strip()
roll = input("Enter Roll No: ").strip()
branch = input("Enter Branch: ").strip()

# ===== SETTINGS =====
BASE_DIR = r"C:\Users\Amgothu Saraswathi\Desktop\data sets CSE3rd year\faces"
MODEL_DIR = r"C:\Users\Amgothu Saraswathi\Desktop\FaceModel"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

person_name = roll
num_photos = 10

person_folder = os.path.join(BASE_DIR, person_name)
os.makedirs(person_folder, exist_ok=True)

# ===== SAVE STUDENT DETAILS =====
DETAILS_JSON = os.path.join(MODEL_DIR, "details.json")

student_data = {}
if os.path.exists(DETAILS_JSON):
    with open(DETAILS_JSON, "r") as f:
        student_data = json.load(f)

student_data[person_name] = {
    "name": name,
    "roll": roll,
    "branch": branch
}

with open(DETAILS_JSON, "w") as f:
    json.dump(student_data, f, indent=4)

# ===== AUTO FACE CAPTURE WITH TIME DELAY =====
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
count = 0

# 🔥 DELAY SETTINGS
last_capture_time = 0
delay = 3  # seconds gap between photos

print(f"[INFO] Capturing faces for {name} ({roll}) ...")
print("[INFO] Auto Capture ON... press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_time = time.time()

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(roi_gray, (200, 200))

        # ⏳ TIME GAP CHECK
        if current_time - last_capture_time >= delay:
            count += 1
            file_path = os.path.join(
                person_folder, f"{person_name}_{count}.jpg"
            )
            cv2.imwrite(file_path, face_resized)
            last_capture_time = current_time

            print(f"[INFO] Saved image {count}")

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Counter display
        cv2.putText(frame, f"{count}/{num_photos}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        # Countdown display
        remaining = max(0, int(delay - (time.time() - last_capture_time)))
        cv2.putText(frame, f"Next capture in: {remaining}s",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

        if count >= num_photos:
            break

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= num_photos:
        break

cap.release()
cv2.destroyAllWindows()

print(f"[INFO] Captured {count} photos ✅ Saved in: {person_folder}")

# ===== TRAINING =====
print("[INFO] Training model...")

model_path = os.path.join(MODEL_DIR, "trainer.yml")
labels_path = os.path.join(MODEL_DIR, "labels.pickle")

recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
x_train = []
y_labels = []

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith(("png", "jpg", "jpeg")):
            path = os.path.join(root, file)
            label = os.path.basename(root)

            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces = face_cascade.detectMultiScale(img, 1.2, 5)
            for (x, y, w, h) in faces:
                roi = img[y:y+h, x:x+w]
                roi = cv2.resize(roi, (200, 200))
                x_train.append(roi)
                y_labels.append(id_)

recognizer.train(x_train, np.array(y_labels))
recognizer.save(model_path)

with open(labels_path, "wb") as f:
    pickle.dump(label_ids, f)

print("[INFO] ✅ Training Complete!")
print(f"[INFO] Model saved at: {model_path}")
print(f"[INFO] Labels saved at: {labels_path}")
print(f"[INFO] Student details saved in: {DETAILS_JSON}")