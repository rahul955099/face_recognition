first install this

!pip install mtcnn opencv-python



Next Preprocessing Code

import cv2
import os
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()

INPUT_DIR = "/content/drive/MyDrive/dataset/dataset/train"
OUTPUT_DIR = "/content/drive/MyDrive/dataset/dataset/preprocessed"
IMG_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)

for person in os.listdir(INPUT_DIR):
    person_path = os.path.join(INPUT_DIR, person)
    save_person_path = os.path.join(OUTPUT_DIR, person)
    os.makedirs(save_person_path, exist_ok=True)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)

        if len(faces) == 0:
            print(f"No face found in {img_name}")
            continue

        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)

        face = img_rgb[y:y+h, x:x+w]

        # ---- Make square ----
        size = max(w, h)
        square_face = np.zeros((size, size, 3), dtype=np.uint8)

        square_face[:face.shape[0], :face.shape[1]] = face

        # Resize
        face_resized = cv2.resize(square_face, (IMG_SIZE, IMG_SIZE))

        save_path = os.path.join(save_person_path, img_name)
        cv2.imwrite(save_path, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))

print("Face preprocessing completed")
