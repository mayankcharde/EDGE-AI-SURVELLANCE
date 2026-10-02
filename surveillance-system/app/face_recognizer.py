"""
Universal Image Recognizer — robust multi-subject edition.

Architecture:
  Stage 1 — Deep Learning Person Re-Identification (ReID) using ResNet18 embeddings
             (matches whole body, costume, and overall appearance).
  Stage 2 — AlchemyFace (YuNet+SFace) for human faces only (optional fallback).
"""
import os
import pickle
import logging
import cv2
import numpy as np
from collections import defaultdict
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

from config import KNOWN_FACES_DIR, FACE_DB_PATH, FACE_COSINE_THRESHOLD

logger = logging.getLogger(__name__)

REID_COSINE_THRESHOLD = 0.70  # Tunable threshold for person ReID cosine similarity

class FaceRecognizer:
    """
    Two-stage universal recognizer.

    Stage 1 — Deep ReID (works for body, costume, overall appearance)
    Stage 2 — AlchemyFace fallback for human faces
    """

    def __init__(self):
        self.threshold = FACE_COSINE_THRESHOLD
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ── Deep ReID Engine (Replaces ORB) ──
        logger.info(f"Loading Person ReID model on {self.device}...")
        self._reid_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self._reid_model.fc = torch.nn.Identity() # Strip classification head to get embeddings
        self._reid_model = self._reid_model.to(self.device)
        self._reid_model.eval()
        
        # Standard ImageNet normalization for ResNet
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)), # ReID standard size (H, W)
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # enrolled: { label: [embedding_vector, ...] }
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
            logger.warning(f"AlchemyFace not available ({e}). ReID-only mode.")

        # ── Load ReID DB ──
        reid_db = self._reid_db_path()
        if os.path.exists(reid_db):
            try:
                with open(reid_db, "rb") as f:
                    self._enrolled = pickle.load(f)
                logger.info(f"Loaded ReID DB: {list(self._enrolled.keys())}")
            except Exception as e:
                logger.warning(f"Could not load ReID DB: {e}")
        else:
            logger.info("No ReID DB — run enroll_faces.py first.")

    # ──────────────────────────────────────────
    # Paths
    # ──────────────────────────────────────────
    @staticmethod
    def _reid_db_path() -> str:
        return FACE_DB_PATH.replace(".pkl", "_reid.pkl")

    # ──────────────────────────────────────────
    # Persist
    # ──────────────────────────────────────────
    def save_db(self):
        os.makedirs(os.path.dirname(FACE_DB_PATH), exist_ok=True)
        with open(self._reid_db_path(), "wb") as f:
            pickle.dump(self._enrolled, f)
        logger.info(f"Saved ReID DB → {self._reid_db_path()}")
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

            if self._enroll_reid(label, img):
                count += 1
                logger.info(f"[ReID] Enrolled '{label}' ← {fname}")
            else:
                logger.warning(f"[ReID] Failed to extract features for {fname}")

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

    def _extract_feature(self, img: np.ndarray) -> np.ndarray:
        """Extracts a normalized 512-d feature vector for a person crop."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self._reid_model(tensor)
            feat = feat / feat.norm(p=2, dim=1, keepdim=True)  # L2 normalize
        return feat.cpu().numpy()[0]

    def _enroll_reid(self, label: str, img: np.ndarray) -> bool:
        """Extract Deep ReID descriptors from an image."""
        try:
            feat = self._extract_feature(img)
            if label not in self._enrolled:
                self._enrolled[label] = []
            self._enrolled[label].append(feat)
            return True
        except Exception as e:
            logger.warning(f"ReID feature extraction failed: {e}")
            return False

    # ──────────────────────────────────────────
    # ReID matching — Cosine Similarity
    # ──────────────────────────────────────────
    def _reid_match(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Calculates cosine similarity with enrolled ReID embeddings.
        Returns (label, confidence) or ("Unknown", 0.0).
        """
        if not self._enrolled:
            return "Unknown", 0.0

        try:
            query_feat = self._extract_feature(crop)
        except Exception:
            return "Unknown", 0.0

        best_label = "Unknown"
        best_score = -1.0

        for label, feats in self._enrolled.items():
            for ref_feat in feats:
                # Cosine similarity since vectors are L2-normalized
                score = float(np.dot(query_feat, ref_feat))
                if score > best_score:
                    best_score = score
                    best_label = label

        if best_score > REID_COSINE_THRESHOLD:
            return best_label, best_score

        return "Unknown", 0.0

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

        crop = frame[y1:y2, x1:x2]

        # ── Stage 1: Deep ReID (Whole Body / Costume) ──
        name, conf = self._reid_match(crop)
        if name != "Unknown":
            logger.debug(f"[ReID] matched '{name}' conf={conf:.2f}")
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