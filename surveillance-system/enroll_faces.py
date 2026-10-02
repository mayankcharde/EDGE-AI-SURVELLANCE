"""
Universal face/subject enrollment script.

Supports ANY subject (humans, animals, objects).
Place one image per subject in data/known_faces/<name>.jpg, then run:

    python enroll_faces.py

The script will:
  1. Extract ORB features from EVERY enrolled image (works on anything)
  2. Also run AlchemyFace enrollment for human faces (if available)
  3. Save both DBs to disk so the pipeline can load them on startup
"""
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("enroll")


def main():
    from config import KNOWN_FACES_DIR

    print("=" * 50)
    print("  Universal Subject Enrollment")
    print("=" * 50)

    # Check folder
    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"\n[ERROR] Known faces folder not found: {KNOWN_FACES_DIR}")
        sys.exit(1)

    images = [
        f for f in os.listdir(KNOWN_FACES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if not images:
        print(f"\n[!] No images found in {KNOWN_FACES_DIR}")
        print("    → Add images named <subject_name>.jpg and re-run.")
        sys.exit(0)

    print(f"\nFound {len(images)} image(s) to enroll: {images}\n")

    from app.face_recognizer import FaceRecognizer
    fr = FaceRecognizer()
    n  = fr.enroll_from_folder()

    print()
    if n == 0:
        print("[!] No subjects were enrolled.")
        print("    Make sure each image has clear, non-blurry content.")
    else:
        print(f"[✓] Successfully enrolled {n} subject(s).")
        print(f"    Saved to:  data/face_db_orb.pkl")
        print(f"    Run now:   python run.py")


if __name__ == "__main__":
    main()