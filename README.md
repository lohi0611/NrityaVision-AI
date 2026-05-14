# 🩰 NrityaVision AI
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-v2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-00A6ED?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)

**NrityaVision AI** is a state-of-the-art Bharatanatyam Mudra Recognition system. It bridges classical Indian dance with modern Computer Vision to help students and enthusiasts learn and refine their hand gestures (Asamyuta and Samyuta Hastas).

---

## ✨ Features
- 🚀 **Real-time Detection**: Millisecond-latency detection using MediaPipe's latest Hand Landmarker API.
- 🧠 **ML-Powered**: Random Forest Classifier trained on a custom dataset of 50 classical mudras.
- 🎨 **Cultural Context**: Displays not just the name, but the **meaning** and **traditional usage** (Viniyoga) of each mudra.
- 🎥 **Dual-Hand Support**: Capable of detecting and classifying gestures from both hands simultaneously.
- 📊 **Confidence Visualization**: Real-time confidence tracking for precise learning.

---

## 🛠️ Tech Stack
- **Frontend**: HTML5, Vanilla CSS3 (Custom Glassmorphism Design)
- **Backend**: Flask (Python)
- **Vision**: OpenCV & MediaPipe
- **Machine Learning**: Scikit-Learn, Pandas, NumPy
- **Dataset Source**: Hugging Face Datasets

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/NrityaVision-AI.git
cd NrityaVision-AI
```

### 2. Set up Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your browser.

---

## 📂 Project Structure
- `app.py`: Main Flask application handling the webcam stream and prediction logic.
- `train_model.py`: Script to train the Random Forest model on landmark data.
- `mudra_info.py`: Database containing cultural meanings and usages for 50 mudras.
- `extract_landmarks.py`: Utility to process the dataset into landmark features.
- `static/`: CSS and styling assets.
- `templates/`: HTML templates.

---

## 📜 Mudra Database
The system currently recognizes 50 distinct mudras, including:
- **Asamyuta Hastas**: Pataka, Tripataka, Ardhapataka, etc.
- **Samyuta Hastas**: Anjali, Kapota, Karkata, etc.

---

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.

---
*Developed with ❤️ for Indian Classical Arts.*
