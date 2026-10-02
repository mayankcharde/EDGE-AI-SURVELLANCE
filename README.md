<div align="center">
  <h1>🛡️ Edge AI Surveillance System</h1>
  <p><strong>Intelligent, Centralized Video Analytics & Real-Time Monitoring</strong></p>

  [![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
  [![YOLOv8](https://img.shields.io/badge/YOLOv8-8.1.29-yellow.svg)](https://ultralytics.com/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

---

## 📖 Overview

The **Edge AI Surveillance System** is a powerful, centralized video analytics pipeline designed for real-time person detection, facial recognition, and abnormal behavior monitoring. Built with high-performance frameworks like YOLOv8 and FastAPI, it ingests video streams (RTSP or webcam), processes them using advanced computer vision techniques, and serves a live, real-time dashboard. 

The system goes beyond standard detection by offering **explainable alerts** (XAI) and comprehensive event logging, ensuring that every automated decision is transparent and auditable.

---

## ✨ Key Features

- 🎯 **Advanced Person Detection & Tracking**: Utilizes YOLOv8 and ByteTrack for robust, high-speed multi-person tracking.
- 👤 **Facial Recognition**: Seamlessly identifies known individuals using state-of-the-art embedding models.
- 🚨 **Abnormal Behavior Detection**:
  - **Restricted-Zone Intrusion**: Triggers alerts when unauthorized entities enter defined polygons.
  - **Loitering Detection**: Identifies individuals dwelling in specific areas beyond a set time limit.
  - **Running & Fall Detection**: Uses aspect-ratio heuristics and speed tracking for safety monitoring.
  - **Unusual Movement (Wandering)**: Detects erratic or non-standard movement patterns.
- 🧠 **Explainable AI (XAI)**: Rule-trace based explanations and structured feature attribution for all generated alerts.
- 📊 **Real-Time Dashboard**: An interactive, FastAPI and Vanilla JS powered web interface for monitoring streams, events, and system health.
- 🔔 **Multi-Channel Notifications**: Automated Email and Telegram alerts with attached snapshot evidence.
- 💾 **Reliable Event Logging**: SQLite-based event storage, complete with frame snapshots for historical review.

---

## 🛠️ Technology Stack

- **Core & Backend**: Python 3.8+, FastAPI, Uvicorn
- **Computer Vision**: OpenCV, Ultralytics (YOLOv8), PyTorch, TorchVision
- **Face Recognition**: `alchemyface` / `dlib` / `face-recognition`
- **Frontend**: HTML5, CSS3, Vanilla JavaScript, Jinja2
- **Database**: SQLite (via standard Python `sqlite3`)
- **Messaging/Alerts**: Telegram Bot API (`python-telegram-bot`), SMTP (`smtplib`)

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8 - 3.11** (Recommended)
- **Git**
- A working webcam or an **RTSP stream URL**

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-username/surveillance-system.git
cd surveillance-system

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create your environment file by copying the provided example:

```bash
cp .env.example .env
```

Edit the `.env` file to configure:
- RTSP Stream URLs
- Telegram Bot Token & Chat ID
- Email SMTP Credentials
- Detection Thresholds

### 3. Enroll Faces (Optional)

To enable facial recognition, create a directory for known faces and run the enrollment script:

```bash
mkdir -p data/known_faces
# Place 1 image per person in data/known_faces/ formatted as <Name>.jpg
python enroll_faces.py
```

### 4. Run the System

Start the FastAPI server and the video processing pipeline:

```bash
python run.py
```

- The system will automatically attempt to open the dashboard in your default browser.
- **Dashboard URL**: [http://localhost:8000](http://localhost:8000)
- **Note**: The camera pipeline remains idle until a source is selected via the web UI.

---

## ⚙️ Advanced Tuning

- **Zone Configuration**: Edit `config.py` to define custom polygonal coordinates for restricted zones.
- **Model Upgrades**: For improved accuracy on powerful machines, swap the default `yolov8n.pt` with larger models like `yolov8s.pt` or `yolov8x.pt`.
- **Headless Mode**: For server deployments, ensure `show_window=False` is set in `run.py`.

---

## 📁 Project Structure

```text
surveillance-system/
├── app/                  # Core application logic
│   ├── alerts.py         # Telegram/Email notification logic
│   ├── behavior.py       # Heuristics for loitering, falls, etc.
│   ├── camera.py         # Stream ingestion and frame reading
│   ├── database.py       # SQLite event logging
│   ├── detector.py       # YOLOv8 integration
│   ├── face_recognizer.py# Facial feature extraction & matching
│   ├── main.py           # FastAPI application & routes
│   ├── pipeline.py       # The central video processing loop
│   └── xai.py            # Explainability and rule-tracing
├── data/                 # Local data storage (faces, events, models)
├── static/               # CSS, JS, and static web assets
├── templates/            # Jinja2 HTML templates for the dashboard
├── .env.example          # Template for environment variables
├── config.py             # Global configuration and zone definitions
├── enroll_faces.py       # Utility to compute face embeddings
├── requirements.txt      # Python dependencies
└── run.py                # Main entry point to launch the system
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---
*Built with ❤️ for intelligent and transparent security monitoring.*
