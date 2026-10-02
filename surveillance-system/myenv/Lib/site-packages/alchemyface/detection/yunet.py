"""YuNet face detection through OpenCV's DNN runtime.

OpenCV hands back an ``(N, 15) float32`` array — bounding box in columns 0-3,
five landmarks in 4-13, score in 14 — or ``None`` when nothing is found. The
box is not clamped to the image, so it can start at a negative coordinate or
run off the right edge; ``face_from_row`` fixes that before anyone slices an
array with it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from alchemyface.models import DETECTOR, resolve
from alchemyface.types import Face

_ROW_WIDTH = 15


def face_from_row(row: NDArray[np.float32], image_shape: tuple[int, ...]) -> Face:
    """Convert one OpenCV detection row into a :class:`Face`, clamped."""
    height, width = int(image_shape[0]), int(image_shape[1])
    x = max(0, int(row[0]))
    y = max(0, int(row[1]))
    w = max(0, min(int(row[2]), width - x))
    h = max(0, min(int(row[3]), height - y))
    return Face(
        bbox=(x, y, w, h),
        landmarks=np.asarray(row[4:14], dtype=np.float32).reshape(5, 2),
        confidence=float(row[14]),
    )


def row_from_face(face: Face) -> NDArray[np.float32]:
    """Rebuild the 15-column row OpenCV needs for ``alignCrop``."""
    row = np.zeros(_ROW_WIDTH, dtype=np.float32)
    row[:4] = face.bbox
    row[4:14] = face.landmarks.reshape(-1)
    row[14] = face.confidence
    return row


MIN_SCORE = 0.05
MAX_SCORE = 0.99
"""The range OpenCV's YuNet behaves sensibly over.

A score is a probability, so values outside it are meaningless: at 0 every
anchor is a face and at 1 nothing ever is. cv2 accepts them without complaint
and then returns garbage, so they are clamped here rather than at the caller —
the GUI, the CLI and a library user all reach the same floor.
"""


def clamp_score(value: float) -> float:
    """``value`` confined to the range YuNet behaves over. NaN yields the default."""
    number = float(value)
    if number != number:  # NaN compares unequal to itself
        return 0.9
    return max(MIN_SCORE, min(MAX_SCORE, number))


class YuNetDetector:
    """Implements :class:`~alchemyface.detection.base.Detector` with YuNet."""

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        model_dir: Path | str | None = None,
        score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        score_threshold = clamp_score(score_threshold)
        path = Path(model_path) if model_path else resolve(DETECTOR, model_dir)
        # The `Xxx.create` class-method form is what cv2's bundled type stubs
        # declare; the module-level `FaceDetectorYN_create` alias is not, and
        # trips mypy with "Module has no attribute".
        self._detector = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score_threshold, nms_threshold, top_k)
        self._score_threshold = float(score_threshold)
        self._input_size = (320, 320)

    @property
    def score_threshold(self) -> float:
        return self._score_threshold

    def set_score_threshold(self, value: float) -> None:
        """Change the detection score without rebuilding the network.

        Rebuilding would discard nothing here, but the caller may hold cached
        embeddings that the threshold does not invalidate.
        """
        self._score_threshold = clamp_score(value)
        self._detector.setScoreThreshold(self._score_threshold)

    def detect(self, image: NDArray[np.uint8]) -> list[Face]:
        """Every face found in a BGR image, with boxes clamped to its bounds."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected a three-channel BGR image, got shape {image.shape}")
        height, width = image.shape[:2]
        # The network is built for a fixed input size; it must be told
        # whenever the frame dimensions change or OpenCV raises.
        if (width, height) != self._input_size:
            self._detector.setInputSize((width, height))
            self._input_size = (width, height)

        _, rows = self._detector.detect(image)
        if rows is None:
            return []
        # Iterating a 2-D ndarray yields rows, but the numpy stubs type the
        # element as a scalar, so each row is re-asserted as a float32 array.
        return [face_from_row(np.asarray(row, dtype=np.float32), image.shape) for row in rows]
