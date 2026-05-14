import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import easyocr
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mudra_info import mudra_info

# --- CONFIG ---
VIDEO_PATH = "TEST VIDEO.mp4"
CSV_NAME = "mudra_landmarks.csv"
MODEL_PATH = "hand_landmarker.task"
VALID_MUDRAS = list(mudra_info.keys())

def get_best_match(text):
    text = text.lower().strip().replace("aa", "a")
    if not text: return None
    
    # Mapping for common variations
    variations = {
        "pataka": "Pathaka",
        "tripataka": "Tripathaka",
        "ardhapataka": "Ardhapathaka",
        "kartareemukha": "Katrimukha",
        "kartarimukha": "Katrimukha",
        "shikhara": "Sikharam",
        "kapitha": "Kapith",
        "alapadma": "Alapadmam"
    }
    
    if text in variations:
        return variations[text]

    # Simple matching
    for m in VALID_MUDRAS:
        m_low = m.lower()
        if text in m_low or m_low in text:
            return m
    return None

def smart_extract():
    print("Initializing Smart OCR Extractor...", flush=True)
    reader = easyocr.Reader(['en'], gpu=False) # GPU=False for compatibility
    
    # MediaPipe Setup
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3, # Lowered for video
        min_hand_presence_confidence=0.3
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(VIDEO_PATH)
    all_data = []
    ts = 0
    current_label = None
    frame_count = 0
    
    print(f"Processing {VIDEO_PATH}...", flush=True)

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame_count += 1
        ts += 33 # Assume 30fps

        # Run OCR every 10 frames for better accuracy
        if frame_count % 10 == 0:
            results = reader.readtext(frame)
            for (bbox, text, prob) in results:
                print(f"Frame {frame_count} | Found Text: '{text}' (prob: {prob:.2f})", flush=True)
                match = get_best_match(text)
                if match:
                    if match != current_label:
                        print(f"MATCH! Setting label to: {match}", flush=True)
                    current_label = match

        if current_label:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_image, ts)

            if result.hand_landmarks:
                landmarks = []
                for hand_lms in result.hand_landmarks[:2]:
                    for lm in hand_lms:
                        landmarks.extend([lm.x, lm.y, lm.z])
                
                while len(landmarks) < 126:
                    landmarks.append(0)
                
                landmarks.append(current_label)
                all_data.append(landmarks)

        # Save progress every 100 frames
        if frame_count % 100 == 0 and all_data:
            df = pd.DataFrame(all_data)
            file_exists = os.path.isfile(CSV_NAME)
            df.to_csv(CSV_NAME, mode='a', header=not file_exists, index=False)
            print(f"Incremental Save: Added {len(all_data)} samples at frame {frame_count}", flush=True)
            all_data = [] # Clear buffer after saving

    cap.release()

    if all_data: # Save any remaining data
        df = pd.DataFrame(all_data)
        file_exists = os.path.isfile(CSV_NAME)
        df.to_csv(CSV_NAME, mode='a', header=not file_exists, index=False)
        print(f"Final Save: Added {len(all_data)} remaining samples.")
        return True
    
    return os.path.isfile(CSV_NAME) # Return true if we have any data at all

if __name__ == "__main__":
    if smart_extract():
        print("Re-training model with new data...")
        import subprocess
        subprocess.run([".\\.venv\\Scripts\\python.exe", "train_model.py"])
