# import os
# from dotenv import load_dotenv

# load_dotenv()


# # ---------- Source selection ----------
# # SOURCE_TYPE: "webcam" | "rtsp" | "video"
# SOURCE_TYPE = os.getenv("SOURCE_TYPE", "webcam").lower()
# WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX", 0))       # 0 = default laptop webcam
# RTSP_URL = os.getenv("RTSP_URL", "")                    # leave blank if using webcam
# VIDEO_PATH = os.getenv("VIDEO_PATH", "")                # default demo video (optional)

# # Folder for uploaded demo videos
# UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # ---------- Camera ----------
# RTSP_URL = os.getenv("RTSP_URL", "0")  # "0" for webcam
# CAMERA_NAME = os.getenv("CAMERA_NAME", "Camera 1")

# # ---------- Paths ----------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(BASE_DIR, "data")
# SNAPSHOT_DIR = os.path.join(DATA_DIR, "snapshots")
# CLIP_DIR = os.path.join(DATA_DIR, "clips")
# KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
# FACE_DB_PATH = os.path.join(DATA_DIR, "face_db.pkl")
# DB_PATH = os.path.join(DATA_DIR, "events.db")
# LOG_DIR = os.path.join(BASE_DIR, "logs")

# # ---------- Detection ----------
# YOLO_WEIGHTS = "yolov8n.pt"          # change to yolov8s.pt / yolov8x.pt for better accuracy
# PERSON_CLASS_ID = 0
# CONFIDENCE_THRESHOLD = 0.45
# FACE_MATCH_TOLERANCE = 0.5
# FACE_RECOGNITION_EVERY_N_FRAMES = 5

# # ---------- Restricted zone (x1, y1, x2, y2) in pixels ----------
# RESTRICTED_ZONE = (200, 200, 600, 600)

# # ---------- Behavior thresholds ----------
# LOITERING_SECONDS = int(os.getenv("LOITERING_SECONDS", 10))
# RUNNING_SPEED_THRESHOLD = float(os.getenv("RUNNING_SPEED_THRESHOLD", 80))
# FALL_ASPECT_RATIO = float(os.getenv("FALL_ASPECT_RATIO", 1.2))
# DIRECTION_CHANGE_THRESHOLD = int(os.getenv("DIRECTION_CHANGE_THRESHOLD", 5))
# DIRECTION_CHANGE_WINDOW = 5  # seconds
# INTRUSION_COOLDOWN = int(os.getenv("INTRUSION_COOLDOWN", 30))  # seconds between repeat alerts

# # ---------- Email ----------
# EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
# SMTP_SERVER = os.getenv("SMTP_SERVER", "")
# SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
# SMTP_USER = os.getenv("SMTP_USER", "")
# SMTP_PASS = os.getenv("SMTP_PASS", "")
# ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")

# # ---------- Telegram ----------
# TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
# TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
# TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# # ---------- Server ----------
# HOST = "0.0.0.0"
# PORT = 8000


import os
from dotenv import load_dotenv

load_dotenv()

# ---------- Paths (define these FIRST) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SNAPSHOT_DIR = os.path.join(DATA_DIR, "snapshots")
CLIP_DIR = os.path.join(DATA_DIR, "clips")
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
FACE_DB_PATH = os.path.join(DATA_DIR, "face_db.pkl")
DB_PATH = os.path.join(DATA_DIR, "events.db")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Create required directories at import time
for _d in (DATA_DIR, SNAPSHOT_DIR, CLIP_DIR, KNOWN_FACES_DIR, UPLOAD_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------- Source selection ----------
# SOURCE_TYPE: "webcam" | "rtsp" | "video" | "none"
SOURCE_TYPE = os.getenv("SOURCE_TYPE", "none").lower()
WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX", 0))    # 0 = default laptop webcam
RTSP_URL = os.getenv("RTSP_URL", "")                 # leave blank if using webcam
VIDEO_PATH = os.getenv("VIDEO_PATH", "")             # default demo video (optional)

# ---------- Face Recognition (AlchemyFace) ----------
# Cosine similarity threshold for a match.
# 0.363 = SFace's published operating point (recommended default).
# Higher = stricter (fewer false matches, more "Unknown").
FACE_COSINE_THRESHOLD = float(os.getenv("FACE_COSINE_THRESHOLD", 0.363))

# Legacy: used only to derive the cosine threshold if FACE_COSINE_THRESHOLD
# is not set. distance 0.5 → similarity 0.5.
FACE_MATCH_TOLERANCE = 0.5


# ---------- Camera ----------
CAMERA_NAME = os.getenv("CAMERA_NAME", "Camera 1")

# ---------- Detection ----------
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")  # yolov8s.pt / yolov8x.pt for accuracy
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.45
FACE_MATCH_TOLERANCE = 0.5
FACE_RECOGNITION_EVERY_N_FRAMES = 5

# ---------- Restricted zone (x1, y1, x2, y2) in pixels ----------
RESTRICTED_ZONE = (200, 200, 600, 600)

# ---------- Behavior thresholds ----------
# LOITERING_SECONDS = int(os.getenv("LOITERING_SECONDS", 10))
# RUNNING_SPEED_THRESHOLD = float(os.getenv("RUNNING_SPEED_THRESHOLD", 80))
# FALL_ASPECT_RATIO = float(os.getenv("FALL_ASPECT_RATIO", 1.2))
# DIRECTION_CHANGE_THRESHOLD = int(os.getenv("DIRECTION_CHANGE_THRESHOLD", 5))
# DIRECTION_CHANGE_WINDOW = 5                                     # seconds
# INTRUSION_COOLDOWN = int(os.getenv("INTRUSION_COOLDOWN", 30))   # seconds between repeat alerts


# ---------- Behavior Thresholds (perspective-invariant, robust) ----------
#
# RESTRICTED_ZONE: use NORMALIZED coordinates (0.0–1.0) as fractions of the
# frame's width/height. This makes it work with ANY video resolution.
#   (x1, y1, x2, y2)  where all values are 0.0–1.0
#   Example: (0.15, 0.15, 0.55, 0.85) = left 15% to 55% width,
#                                        top 15% to 85% height.
#
# If you want to keep pixel coordinates, set any value > 1.0 and the code
# will auto-detect that and treat them as pixels.
RESTRICTED_ZONE = (0.15, 0.15, 0.55, 0.85)

# --- Speed (in BODY-LENGTHS per second) ---
# Perspective invariant: a person's bbox height shrinks with distance,
# so distance/height is roughly constant regardless of camera position.
#   Walking  ≈ 0.8 – 1.5 body-lengths/sec
#   Jogging  ≈ 1.5 – 2.5
#   Running  ≈ 2.5 – 5+
RUNNING_SPEED_BODY_LENGTHS_PER_SEC = 2.5

# --- Loitering ---
LOITERING_SECONDS = 10
LOITERING_MAX_SPEED_BODY_LENGTHS = 0.4    # must be nearly stationary

# --- Fall (multiple signals required) ---
# A fall is only flagged when:
#   1. Aspect ratio flips from tall (h/w > 1.4) to wide (w/h > 1.3)
#   2. Centroid drops by at least FALL_CENTROID_DROP_FRAC of the frame height
#   3. The wide posture persists for FALL_MIN_DURATION_SEC
FALL_ASPECT_RATIO = 1.3
FALL_STANDING_HW_RATIO = 1.4
FALL_CENTROID_DROP_FRAC = 0.05
FALL_MIN_DURATION_SEC = 1.0

# --- Unusual movement (tortuosity = path length / displacement) ---
#   Straight walk ≈ 1.0
#   Curving      ≈ 1.5
#   Wandering    ≈ 3.0+
UNUSUAL_TORTUOSITY_THRESHOLD = 3.0
UNUSUAL_MIN_DISPLACEMENT_BODY_LENGTHS = 2.0   # ignore if barely moving
UNUSUAL_WINDOW_SEC = 6.0

# --- Robustness ---
SMOOTHING_WINDOW = 7                  # moving-average window for centroid
MIN_HISTORY_FRAMES = 5                # minimum samples before behavior logic
MIN_DETECTION_CONFIDENCE = 0.40       # ignore YOLO detections below this
INTRUSION_COOLDOWN = 20               # seconds between repeat alerts
DEBUG_BEHAVIOR = True                 # print normalized metrics to console

# ---------- Email ----------
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
SMTP_SERVER = os.getenv("SMTP_SERVER", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")

# ---------- Telegram ----------
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ---------- Discord ----------
DISCORD_ENABLED = os.getenv("DISCORD_ENABLED", "false").lower() == "true"
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# ---------- Slack ----------
SLACK_ENABLED = os.getenv("SLACK_ENABLED", "false").lower() == "true"
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# ---------- ntfy.sh ----------
NTFY_ENABLED = os.getenv("NTFY_ENABLED", "false").lower() == "true"
NTFY_SERVER = os.getenv("NTFY_SERVER", "https://ntfy.sh")
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "surveillance-alerts")

# ---------- Server ----------
HOST = "0.0.0.0"
PORT = 8000