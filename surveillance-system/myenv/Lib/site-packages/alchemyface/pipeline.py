"""The facade that sequences detection, embedding and storage.

Recognizer deliberately contains no algorithm. It decides *order* and applies
the *threshold*; everything else is delegated to whichever Detector, Embedder
and FaceStore it was handed. That is what makes a pgvector gallery or a
different embedding model a drop-in rather than a rewrite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from alchemyface.detection.base import Detector
from alchemyface.embedding.base import Embedder
from alchemyface.errors import NoFaceDetectedError
from alchemyface.store.base import FaceStore
from alchemyface.types import Face, Recognition

DEFAULT_THRESHOLD = 0.363
"""SFace's published cosine operating point. A tunable, not a constant:
validate it against your own data before relying on it."""


class Recognizer:
    """Detect, embed, enroll and identify faces."""

    def __init__(
        self,
        *,
        detector: Detector | None = None,
        embedder: Embedder | None = None,
        store: FaceStore | None = None,
        model_dir: Path | str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        """Any component left as ``None`` gets the default implementation.

        Constructing the defaults loads the ONNX weights, downloading them on
        first use. Pass all three components to build a Recognizer that touches
        neither disk nor network.
        """
        # Imported here rather than at module scope so that injecting fakes
        # never pays the cost of importing cv2-backed modules.
        # pylint: disable=import-outside-toplevel
        from alchemyface.detection.yunet import YuNetDetector
        from alchemyface.embedding.sface import SFaceEmbedder
        from alchemyface.store.memory import InMemoryStore

        self.detector: Detector = detector if detector is not None else YuNetDetector(model_dir=model_dir)
        self.embedder: Embedder = embedder if embedder is not None else SFaceEmbedder(model_dir=model_dir)
        self.store: FaceStore = store if store is not None else InMemoryStore(dim=self.embedder.dim)
        self.threshold = threshold

    def detect(self, image: NDArray[np.uint8]) -> list[Face]:
        """Every face in the image."""
        return self.detector.detect(image)

    def embed(self, image: NDArray[np.uint8], face: Face) -> NDArray[np.float32]:
        """The unit-length embedding of one already-detected face."""
        return self.embedder.embed(image, face)

    def enroll(
        self,
        label: str,
        image: NDArray[np.uint8],
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Add the most prominent face in the image to the gallery.

        Raises :class:`NoFaceDetectedError` if there is no face. Enrolling the
        largest face is a deliberate choice: an enrolment photo with a
        bystander in the background should not silently enrol the bystander.
        """
        faces = self.detect(image)
        if not faces:
            raise NoFaceDetectedError(f"no face found in the image supplied for {label!r}")
        face = max(faces, key=lambda candidate: candidate.area)
        return self.store.add(label, self.embed(image, face), metadata)

    def identify(self, image: NDArray[np.uint8]) -> list[Recognition]:
        """One :class:`Recognition` per detected face.

        ``Recognition.match`` is ``None`` when the best candidate falls below
        the threshold, or when the gallery is empty — the caller decides what
        "unknown" should mean rather than the library guessing a label.
        """
        recognitions: list[Recognition] = []
        for face in self.detect(image):
            candidates = self.store.search(self.embed(image, face), k=1)
            best = candidates[0] if candidates else None
            if best is not None and best.score < self.threshold:
                best = None
            recognitions.append(Recognition(face=face, match=best))
        return recognitions
