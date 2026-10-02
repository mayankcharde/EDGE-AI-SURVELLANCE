# AI Surveillance System (Centralized)

Real-time person detection, face recognition, abnormal behavior detection,
and explainable alerts — running on a central server (no edge AI / Raspberry Pi).

## Features
- RTSP/webcam ingestion (multi-threaded)
- Person detection + tracking (YOLOv8 + ByteTrack)
- Face recognition (`face_recognition`/dlib)
- Abnormal behavior detection:
  - Restricted-zone intrusion
  - Loitering
  - Running
  - Fall (aspect-ratio heuristic)
  - Unusual movement (wandering)
- Explainable alerts: rule trace + structured feature attribution
- Real-time dashboard (FastAPI + vanilla JS)
- Email + Telegram alerts
- SQLite event log with snapshots

## Setup

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env         # fill in RTSP URL and credentials
```

## Enroll faces
Place 1 image per person in `data/known_faces/` as `<name>.jpg`, then:
```bash
python enroll_faces.py
```

## Run
```bash
python run.py
```
Open http://localhost:8000 for the dashboard.

## Tuning
Edit `.env` for thresholds, or `config.py` for restricted zone coordinates.

## Notes
- On headless servers, set `show_window=False` in `run.py`.
- For higher accuracy, replace YOLOv8n with `yolov8s.pt`/`yolov8x.pt`
  and use InsightFace instead of dlib.
- XAI is rule-trace based by default. `app/xai.py` includes an optional
  Grad-CAM hook if you plug in a torch classifier.