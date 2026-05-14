import cv2
import pickle
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

# Constants
HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS

with open("mudra_model.pkl", "rb") as f:
    model = pickle.load(f)

# Configure HandLandmarker in LIVE_STREAM mode for webcam
MODEL_PATH = "hand_landmarker.task"

# We use VIDEO mode since we process frame-by-frame with timestamps
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
timestamp_ms = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms += 33  # ~30fps
    result = detector.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        landmarks = []

        for hand_landmarks in result.hand_landmarks[:2]:
            # Draw landmarks on frame
            landmark_list = []
            for lm in hand_landmarks:
                landmark_list.append(
                    NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                )
            
            mp_drawing.draw_landmarks(
                frame, landmark_list, HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            for lm in hand_landmarks:
                landmarks.extend([lm.x, lm.y, lm.z])

        while len(landmarks) < 126:
            landmarks.append(0)

        landmarks = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(landmarks)[0]

        cv2.putText(
            frame,
            f"Mudra: {prediction}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

    cv2.imshow("Bharatanatyam Mudra Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
