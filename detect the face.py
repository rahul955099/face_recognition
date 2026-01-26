#detects the face 
import cv2
import pickle

import json
import os

MODEL_DIR   = r"C:\Users\Amgothu Saraswathi\Desktop\FaceModel"
model_path  = os.path.join(MODEL_DIR, "trainer.yml")
labels_path = os.path.join(MODEL_DIR, "labels.pickle")
details_path= os.path.join(MODEL_DIR, "details.json")

# Load model & metadata
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

with open(labels_path, "rb") as f:
    label_ids = pickle.load(f)           # label_key -> id
# reverse: id -> label_key
id_to_label = {v:k for k,v in label_ids.items()}

with open(details_path, "r", encoding="utf-8") as f:
    people_details = json.load(f)        # label_key -> details

# Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

THRESH = 70  # lower is stricter; try 60..75 depending on your data

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray  = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Preprocess exactly like training
        face_resized = cv2.resize(roi_gray, (200, 200))

        id_, conf = recognizer.predict(face_resized)

        if conf < THRESH and id_ in id_to_label:
            label_key = id_to_label[id_]
            info = people_details.get(label_key, {})
            name   = info.get("name", "karthik")
            roll   = info.get("roll", "4511-23-733-051")
            branch = info.get("branch", "CSE-A")
            color  = (0, 200, 0)  # green
            line1  = f"Name: {name}"
            line2  = f"Roll: {roll}"
            line3  = f"Branch: {branch}"
        else:
            color = (0, 0, 255)   # red
            line1, line2, line3 = "Unknown", "", ""

        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Put text (stack 3 lines)
        cv2.putText(frame, line1, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if line2: cv2.putText(frame, line2, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        if line3: cv2.putText(frame, line3, (x, y+10),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        # Eyes (draw inside face ROI)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

    cv2.imshow("Multi-Person Face & Eye Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        break

cap.release()
cv2.destroyAllWindows()