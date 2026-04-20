import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# =====================================================
# CONFIG
# =====================================================
VIDEO_DIR = r"D:\TDPCL_Code\UTA_VIDEOS\Fold1_part1"
EYE_MODEL_PATH = r"D:\TDPCL_Code\eye_model.h5"

FACE_CASCADE_PATH = r"D:\TDPCL_Code\haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = r"D:\TDPCL_Code\haarcascade_eye.xml"

OUTPUT_CSV = "uta_features.csv"

TARGET_FPS = 3
WINDOW_SECONDS = 10
EYE_THRESHOLD = 0.5

# =====================================================
# LABEL MAP (EDIT IF NEEDED)
# =====================================================
LABEL_MAP = {
    "01": "ALERT",
    "02": "ALERT",
    "03": "ALERT",
    "04": "ALERT",
    "05": "ALERT",
    "06": "ALERT",
    "07": "DROWSY",
    "08": "DROWSY",
    "09": "DROWSY",
    "10": "DROWSY",
    "11": "DROWSY",
    "12": "DROWSY"
}

# =====================================================
# LOAD MODELS
# =====================================================
print("Loading eye model...")
eye_model = tf.keras.models.load_model(EYE_MODEL_PATH)
print("Eye model loaded.")

print("Loading Haar cascades...")
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

print("Face cascade empty:", face_cascade.empty())
print("Eye cascade empty:", eye_cascade.empty())

if face_cascade.empty() or eye_cascade.empty():
    raise RuntimeError("❌ Haar cascade files not loaded. Check file paths.")

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def count_blinks(states):
    blinks = 0
    for i in range(1, len(states)):
        if states[i - 1] == 0 and states[i] == 1:
            blinks += 1
    return blinks

# =====================================================
# MAIN PIPELINE
# =====================================================
rows = []

print("\nStarting video processing...\n")

for subject in sorted(os.listdir(VIDEO_DIR)):
    subject_path = os.path.join(VIDEO_DIR, subject)

    if subject not in LABEL_MAP:
        print(f"Skipping unknown subject: {subject}")
        continue

    label = LABEL_MAP[subject]
    print(f"\n=== Subject {subject} | Label: {label} ===")

    for video in os.listdir(subject_path):
        if not video.lower().endswith(".mp4"):
            continue

        video_path = os.path.join(subject_path, video)
        print(f"\nOpening video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Failed to open video.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(fps // TARGET_FPS))
        frame_id = 0
        eye_states = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(face)

                    if len(eyes) == 0:
                        continue

                    # Use first detected eye only
                    ex, ey, ew, eh = eyes[0]
                    eye_img = face[ey:ey+eh, ex:ex+ew]

                    if eye_img.size == 0:
                        continue

                    eye_img = cv2.resize(eye_img, (24, 24))
                    eye_img = eye_img / 255.0
                    eye_img = eye_img.reshape(1, 24, 24, 1)

                    prob = eye_model.predict(eye_img, verbose=0)[0][0]
                    eye_closed = 1 if prob > EYE_THRESHOLD else 0
                    eye_states.append(eye_closed)

            if frame_id % 100 == 0:
                print(f"  Frame {frame_id}")

            frame_id += 1

        cap.release()

        print("Aggregating windows...")

        window_size = TARGET_FPS * WINDOW_SECONDS

        for i in range(0, len(eye_states) - window_size, window_size):
            window = eye_states[i:i + window_size]

            rows.append({
                "PERCLOS": sum(window) / len(window),
                "BlinkRate": count_blinks(window),
                "Label": label
            })

print("\nProcessing complete.")

# =====================================================
# SAVE CSV
# =====================================================
if len(rows) == 0:
    print("❌ No data extracted. Check videos and cascades.")
else:
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ CSV GENERATED: {OUTPUT_CSV}")
    print("Total rows:", len(df))
