from flask import Flask, render_template, Response, jsonify
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
from mudra_info import mudra_info

app = Flask(__name__)

# Constants
HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS

# Subtle Elegant Colors (BGR)
WARM_GOLD = (130, 200, 230)      # Soft warm gold
SOFT_AMBER = (160, 190, 210)     # Muted amber/cream

# Custom Drawing Specs
custom_landmark_style = mp_drawing.DrawingSpec(
    color=WARM_GOLD, 
    thickness=2, 
    circle_radius=3
)
custom_connection_style = mp_drawing.DrawingSpec(
    color=SOFT_AMBER, 
    thickness=2
)

with open("mudra_model.pkl", "rb") as f:
    model = pickle.load(f)

# Configure HandLandmarker with new Tasks API
MODEL_PATH = "hand_landmarker.task"
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

# Shared state for current prediction
current_prediction = {"label": "Show Mudra", "confidence": 0, "meaning": "", "use": ""}


def generate_frames():
    global current_prediction
    
    # Initialize camera ONLY when a user starts the stream
    camera = cv2.VideoCapture(0)
    timestamp_ms = 0

    try:
        while True:
            success, frame = camera.read()
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
                        frame, 
                        landmark_list, 
                        HAND_CONNECTIONS,
                        landmark_drawing_spec=custom_landmark_style,
                        connection_drawing_spec=custom_connection_style
                    )

                    for lm in hand_landmarks:
                        landmarks.extend([lm.x, lm.y, lm.z])

                while len(landmarks) < 126:
                    landmarks.append(0)

                features = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(features)[0]
                confidence = max(model.predict_proba(features)[0]) * 100

                # Update shared state
                info = mudra_info.get(prediction, {"meaning": "Unknown", "use": "Unknown"})
                current_prediction = {
                    "label": prediction,
                    "confidence": round(confidence, 1),
                    "meaning": info["meaning"],
                    "use": info["use"],
                }
            else:
                current_prediction = {"label": "Show Mudra", "confidence": 0, "meaning": "", "use": ""}

            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
    finally:
        # Ensure camera is released when the stream stops or client disconnects
        camera.release()
        print("🎥 Camera released successfully.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/prediction")
def prediction():
    return jsonify(current_prediction)


if __name__ == "__main__":
    app.run(debug=True)
