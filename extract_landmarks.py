import os
import cv2
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATASET_DIR = "bharatanatyam_dataset/train"
OUTPUT_CSV = "mudra_landmarks.csv"
MODEL_PATH = "hand_landmarker.task"

# Configure the HandLandmarker with the new Tasks API
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

rows = []

for label in sorted(os.listdir(DATASET_DIR)):
    label_path = os.path.join(DATASET_DIR, label)

    if not os.path.isdir(label_path):
        continue

    print("Processing:", label)

    for image_name in os.listdir(label_path):
        image_path = os.path.join(label_path, image_name)
        image = cv2.imread(image_path)

        if image is None:
            continue

        # Convert BGR to RGB and wrap in MediaPipe Image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect(mp_image)

        if result.hand_landmarks:
            landmarks = []

            for hand_landmarks in result.hand_landmarks[:2]:
                for lm in hand_landmarks:
                    landmarks.extend([lm.x, lm.y, lm.z])

            # If only one hand is detected, pad the rest with zeros to maintain 126 features (21*3*2)
            while len(landmarks) < 126:
                landmarks.append(0)

            rows.append([label] + landmarks)

columns = ["label"] + [f"f{i}" for i in range(126)]
df = pd.DataFrame(rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
print("Total samples:", len(df))
