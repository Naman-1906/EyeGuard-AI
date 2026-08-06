# 👁️ EyeGuard AI – Drowsiness Detection System

An AI-powered real-time drowsiness detection system that monitors a user's eyes through a webcam and alerts them when signs of drowsiness are detected. The project is designed to improve safety for drivers, students, and professionals who work for long hours.

---

## 📖 Project Overview

EyeGuard AI uses computer vision techniques to detect a user's face and eyes in real time. It continuously monitors eye closure, and if the eyes remain closed for a specified duration, the system assumes the user is drowsy and immediately triggers both visual and audio alerts.

This application can help reduce accidents caused by fatigue and improve user awareness during prolonged computer usage.

---

## ✨ Features

- 🎥 Real-time webcam monitoring
- 👤 Face detection using Haar Cascade Classifiers
- 👀 Eye detection and tracking
- ⏱️ Detects prolonged eye closure
- 🔔 Visual warning on the screen
- 🔊 Audio alarm to wake the user
- 🧵 Multithreaded alarm system for smooth performance
- 💻 Simple and lightweight desktop application

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| OpenCV | Computer Vision and Webcam Processing |
| NumPy | Numerical Computation |
| Haar Cascade Classifiers | Face and Eye Detection |
| pygame | Audio Alert (Optional) |
| winsound | Windows Audio Fallback |
| threading | Alarm Handling |
| urllib | Automatic Download of Cascade XML Files |
| pathlib & os | File and Path Management |

---

## 📂 Project Structure

```
EyeGuard-AI/
│── eyeguard_ai.py          # Main application
│── haarcascade_frontalface_default.xml
│── haarcascade_eye.xml
│── alarm.wav               # Alarm sound
│── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/EyeGuard-AI.git
cd EyeGuard-AI
```

### 2. Install Dependencies

```bash
pip install opencv-python numpy pygame
```

---

## ▶️ Run the Project

```bash
python eyeguard_ai.py
```

The application will:

- Open the webcam
- Detect the user's face and eyes
- Monitor eye closure
- Trigger an alarm if drowsiness is detected

---

## 🧠 How It Works

1. Capture live video from the webcam.
2. Detect the face using Haar Cascade.
3. Detect both eyes within the face region.
4. Continuously monitor eye visibility.
5. If the eyes remain closed for a predefined duration:
   - Display a warning message.
   - Play an alarm sound.
6. Continue monitoring until the user becomes alert again.

---

## 📌 Applications

- Driver drowsiness detection
- Student study sessions
- Office and remote work monitoring
- Fatigue detection during long computer usage
- Safety monitoring systems

---

## 🚀 Future Improvements

- Deep Learning-based eye detection
- Eye Aspect Ratio (EAR) implementation
- Blink rate analysis
- Head pose estimation
- Mobile application support
- Email/SMS emergency alerts
- Driver monitoring dashboard

---

## 📷 Sample Output
<img width="812" height="655" alt="image" src="https://github.com/user-attachments/assets/8d3016bc-c40d-464c-9806-c06bfd4de10e" />


- Face detected successfully
- Eyes tracked in real time
- Warning displayed when eyes remain closed
- Audio alarm activated automatically

---

## 👨‍💻 Author

**Naman Gupta**

If you found this project useful, consider giving it a ⭐ on GitHub!

---

## 📄 License

This project is intended for educational and learning purposes. Feel free to use and modify it for personal or academic projects.
