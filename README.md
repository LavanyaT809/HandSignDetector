# 🤟 Hand Sign Detector

A Python-based hand gesture recognition system built using **OpenCV** and **MediaPipe**. This project detects and tracks hand landmarks in real time from webcam video feed, laying the foundation for building gesture-controlled applications or sign language interpreters.

---

## 🚀 Features

- 📷 Real-time hand tracking using webcam
- ✋ Detects 21 hand landmarks with high accuracy
- 🧠 Easily extendable for gesture/sign classification
- 🛠️ Clean and modular Python code
- 💡 Great for beginners learning computer vision

---

## 🛠️ Tech Stack

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy

---

## 📦 Installation Guide

Follow the steps below to set up and run the project on your local machine:

### 1. Clone the repository

```bash
git clone https://github.com/LavanyaT809/HandSignDetector.git
cd HandSignDetector
```

### 2.Create a virtual environment
python -m venv venv
- For Linux/Mac
  
  ```bash
source venv/bin/activate
```

- For Windows

  ```bash
venv\Scripts\activate
```

### 3. Install dependencies
Install from requirements.txt:

```bash
pip install -r requirements.txt
```
### 📁 Folder Structure
HandSignDetector/
├── hand_tracker.py        # Main file for running the detector
├── hand_module.py         # Custom module using MediaPipe Hand class
├── requirements.txt       # List of dependencies
├── demo/                  # Folder to store screenshots or demo videos
└── README.md              # Project documentation 

### ✅ Future Improvements
-🔤 Add full A–Z ASL gesture classification

-🧠 Integrate ML model for custom sign recognition

-🖥️ Build GUI using Tkinter or a web app using Streamlit

-🔊 Add text-to-speech output for recognized signs

-🌐 Deploy using Flask or Streamlit for web access

