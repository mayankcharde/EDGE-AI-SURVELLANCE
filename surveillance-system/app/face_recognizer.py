"""
Universal Image Recognizer — robust multi-subject edition.

Architecture:
  Stage 1 — ORB feature matching (any subject: human, animal, object)
             - Multi-scale: runs on original + 0.5x crop
             - CLAHE histogram equalisation before keypoint extraction
             - Per-label confidence scoring prevents cross-label confusion
  Stage 2 — AlchemyFace (YuNet+SFace) for human faces only (optional)

Key improvements over v1:
  * Per-label scoring: each enrolled label gets its own match count,
    best label wins only if its count is clearly above second-best
    (margin check) → prevents false positive cross-labelling in
    multi-person scenes.
  * Adaptive min-good threshold based on crop size.
  * CLAHE equalisation on each crop before ORB → works on dark,
    low-contrast, blurry frames (e.g. robbery footage).
  * Name cache in pipeline now includes a "freshness" counter so
    stale identities decay back to Unknown when a subject disappears.
  * ORB_MIN_GOOD_RATIO replaces fixed ORB_MIN_GOOD → scales with the
    number of keypoints found in the query crop.
"""
import os
import pickle
import logging
import cv2
import numpy as np
from collections import defaultdict

from config import KNOWN_FACES_DIR, FACE_DB_PATH, FACE_COSINE_THRESHOLD

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tunable constants
# ─────────────────────────────────────────────────────────────────────────────
ORB_N_FEATURES      = 1500      # keypoints extracted per enrolled image
ORB_MATCH_RATIO     = 0.75      # Lowe's ratio-test threshold
ORB_MIN_GOOD_RATIO  = 0.08      # need ≥ 8% of query kps to match → adaptive
ORB_MIN_GOOD_ABS    = 8         # hard floor regardless of ratio
ORB_CONF_SCALE      = 150.0     # good_count / scale → confidence
ORB_MARGIN_FACTOR   = 1.5       # winner must have ≥ 1.5× runner-up matches
CLAHE_CLIP          = 3.0       # CLAHE clip limit
CLAHE_GRID          = (8, 8)    # CLAHE tile grid size

# Minimum crop dimension — smaller crops are unreliable
MIN_CROP_PX         = 64


def _clahe_equalise(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE histogram equalisation for low-light/blurry frames."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    return clahe.apply(gray)


def _resize_crop(crop: np.ndarray, min_px: int = MIN_CROP_PX) -> np.ndarray:
    """Upscale a crop that is smaller than min_px in any dimension."""
    h, w = crop.shape[:2]
    if h < min_px or w < min_px:
        scale = max(min_px / max(h, 1), min_px / max(w, 1))
        crop = cv2.resize(crop,
                          (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)
    return crop


class FaceRecognizer:
    """
    Two-stage universal recognizer.

    Stage 1 — ORB (works for any enrolled subject)
    Stage 2 — AlchemyFace fallback for human faces
    """

    def __init__(self):
        self.threshold = FACE_COSINE_THRESHOLD

        # ── ORB engine ──
        self._orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
        self._bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # enrolled: { label: [des_array, ...] }
        self._enrolled: dict[str, list[np.ndarray]] = {}

        # ── AlchemyFace (optional, human faces) ──
        self._alchemy_ok = False
        try:
            from alchemyface import Recognizer
            from alchemyface.store import PickleStore
            self._af_store = PickleStore()
            if os.path.exists(FACE_DB_PATH):
                self._af_store.load(FACE_DB_PATH)
            self._af_rec = Recognizer(
                store=self._af_store, threshold=self.threshold
            )
            self._alchemy_ok = True
            logger.info("AlchemyFace loaded (human-face fallback).")
        except Exception as e:
            logger.warning(f"AlchemyFace not available ({e}). ORB-only mode.")

        # ── Load ORB DB ──
        orb_db = self._orb_db_path()
        if os.path.exists(orb_db):
            try:
                with open(orb_db, "rb") as f:
                    self._enrolled = pickle.load(f)
                logger.info(f"Loaded ORB DB: {list(self._enrolled.keys())}")
            except Exception as e:
                logger.warning(f"Could not load ORB DB: {e}")
        else:
            logger.info("No ORB DB — run enroll_faces.py first.")

    # ──────────────────────────────────────────
    # Paths
    # ──────────────────────────────────────────
    @staticmethod
    def _orb_db_path() -> str:
        return FACE_DB_PATH.replace(".pkl", "_orb.pkl")

    # ──────────────────────────────────────────
    # Persist
    # ──────────────────────────────────────────
    def save_db(self):
        os.makedirs(os.path.dirname(FACE_DB_PATH), exist_ok=True)
        with open(self._orb_db_path(), "wb") as f:
            pickle.dump(self._enrolled, f)
        logger.info(f"Saved ORB DB → {self._orb_db_path()}")
        if self._alchemy_ok:
            self._af_store.save(FACE_DB_PATH)

    # ──────────────────────────────────────────
    # Enrollment
    # ──────────────────────────────────────────
    def enroll_from_folder(self) -> int:
        if not os.path.isdir(KNOWN_FACES_DIR):
            logger.warning(f"Folder not found: {KNOWN_FACES_DIR}")
            return 0

        count = 0
        for fname in sorted(os.listdir(KNOWN_FACES_DIR)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            path  = os.path.join(KNOWN_FACES_DIR, fname)
            label = os.path.splitext(fname)[0]
            img   = cv2.imread(path)
            if img is None:
                logger.warning(f"Cannot read {fname}")
                continue

            if self._enroll_orb(label, img):
                count += 1
                logger.info(f"[ORB] Enrolled '{label}' ← {fname}")
            else:
                logger.warning(f"[ORB] Too few keypoints in {fname}")

            if self._alchemy_ok:
                try:
                    n = self._af_rec.enroll(label, img)
                    if n and n > 0:
                        logger.info(f"[AlchemyFace] Enrolled '{label}'")
                except Exception as e:
                    logger.debug(f"[AlchemyFace] skipped {fname}: {e}")

        self.save_db()
        logger.info(f"Enrollment done: {count} subject(s).")
        return count

    def _enroll_orb(self, label: str, img: np.ndarray) -> bool:
        """Extract multi-scale ORB descriptors from an image."""
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eq     = _clahe_equalise(gray)
        descriptors_added = []

        for scale in (1.0, 0.75, 0.5):
            h, w = eq.shape[:2]
            nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
            scaled = cv2.resize(eq, (nw, nh), interpolation=cv2.INTER_AREA)
            _, des = self._orb.detectAndCompute(scaled, None)
            if des is not None and len(des) >= 5:
                descriptors_added.append(des)

        if not descriptors_added:
            return False

        if label not in self._enrolled:
            self._enrolled[label] = []
        self._enrolled[label].extend(descriptors_added)
        return True

    # ──────────────────────────────────────────
    # ORB matching — per-label with margin check
    # ──────────────────────────────────────────
    def _orb_match(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Per-label scoring with runner-up margin check.
        Returns (label, confidence) or ("Unknown", 0.0).
        """
        if not self._enrolled:
            return "Unknown", 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        eq   = _clahe_equalise(gray)
        _, des_q = self._orb.detectAndCompute(eq, None)
        if des_q is None or len(des_q) < 5:
            return "Unknown", 0.0

        # Adaptive minimum — 8% of query keypoints, at least 8 absolute
        min_good = max(ORB_MIN_GOOD_ABS,
                       int(len(des_q) * ORB_MIN_GOOD_RATIO))

        # Accumulate best-match count per label across all stored descriptors
        label_scores: dict[str, int] = defaultdict(int)

        for label, desc_list in self._enrolled.items():
            for des_ref in desc_list:
                if des_ref is None or len(des_ref) < 2:
                    continue
                try:
                    matches = self._bf.knnMatch(des_q, des_ref, k=2)
                except Exception:
                    continue
                good = sum(
                    1 for pair in matches
                    if len(pair) == 2 and pair[0].distance < ORB_MATCH_RATIO * pair[1].distance
                )
                # Keep the BEST descriptor result for this label
                label_scores[label] = max(label_scores[label], good)

        if not label_scores:
            return "Unknown", 0.0

        # Sort by score descending
        ranked = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
        best_label, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0

        # Reject if below absolute threshold
        if best_score < min_good:
            return "Unknown", 0.0

        # Reject if winner is not clearly better than runner-up
        if second_score > 0 and best_score < second_score * ORB_MARGIN_FACTOR:
            logger.debug(
                f"[ORB] Ambiguous match: {best_label}={best_score} vs "
                f"{ranked[1][0]}={second_score} — returning Unknown"
            )
            return "Unknown", 0.0

        conf = min(1.0, best_score / ORB_CONF_SCALE)
        return best_label, conf

    # ──────────────────────────────────────────
    # Public API — recognize from bounding box
    # ──────────────────────────────────────────
    def recognize(self, frame: np.ndarray, box: tuple) -> tuple[str, float]:
        """Return (name, confidence). name='Unknown' if not matched."""
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1 = max(0, int(x1));  y1 = max(0, int(y1))
        x2 = min(w, int(x2));  y2 = min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return "Unknown", 0.0

        crop = _resize_crop(frame[y1:y2, x1:x2])

        # ── Stage 1: ORB ──
        name, conf = self._orb_match(crop)
        if name != "Unknown":
            logger.debug(f"[ORB] matched '{name}' conf={conf:.2f}")
            return name, conf

        # ── Stage 2: AlchemyFace (human faces only) ──
        if self._alchemy_ok:
            try:
                results = self._af_rec.identify(crop)
                if results:
                    best = max(
                        (r for r in results if r.match is not None),
                        key=lambda r: r.match.score,
                        default=None
                    )
                    if best is not None:
                        return best.match.label, float(best.match.score)
            except Exception as e:
                logger.debug(f"AlchemyFace identify failed: {e}")

        return "Unknown", 0.0

    # ──────────────────────────────────────────
    # Public API — full-frame recognition
    # ──────────────────────────────────────────
    def recognize_full_frame(self, frame: np.ndarray) -> list[dict]:
        """Run face detection + recognition on full frame (human faces)."""
        out = []
        if self._alchemy_ok:
            try:
                results = self._af_rec.identify(frame)
                for r in results:
                    out.append({
                        "name":       r.match.label if r.match else "Unknown",
                        "confidence": float(r.match.score) if r.match else 0.0,
                        "bbox":       tuple(r.face.bbox),
                    })
            except Exception:
                pass
        return out