"""AlchemyFace — face detection and recognition on YuNet and SFace.

The public surface is deliberately small::

    from alchemyface import Recognizer

    r = Recognizer()
    r.enroll("prashant", image)
    r.identify(frame)

Everything else is a seam. ``Detector``, ``Embedder`` and ``FaceStore`` are
protocols, so any conforming object can be substituted without touching the
pipeline.
"""

from alchemyface.errors import (
    AlchemyFaceError,
    ModelDownloadError,
    ModelNotFoundError,
    NoFaceDetectedError,
    PickleSchemaError,
)
from alchemyface.pipeline import DEFAULT_THRESHOLD, Recognizer
from alchemyface.types import Face, Match, Recognition, StoreEntry

__version__ = "1.2.0"

__all__ = [
    "AlchemyFaceError",
    "DEFAULT_THRESHOLD",
    "Face",
    "Match",
    "ModelDownloadError",
    "ModelNotFoundError",
    "NoFaceDetectedError",
    "PickleSchemaError",
    "Recognition",
    "Recognizer",
    "StoreEntry",
    "__version__",
]
