"""Value types shared across AlchemyFace. NumPy is the only import."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, eq=False)
class Face:
    """A detected face: where it is, how to align it, how sure the detector was.

    ``eq=False`` is deliberate. ``landmarks`` is an ndarray, and a generated
    ``__eq__`` would compare element-wise and return an array, so any ``==``
    between two Faces would raise "truth value of an array is ambiguous".
    Identity comparison is the useful default here.
    """

    bbox: tuple[int, int, int, int]
    """Pixel bounding box as ``(x, y, width, height)``, clamped to the image."""

    landmarks: NDArray[np.float32]
    """``(5, 2)`` array: right eye, left eye, nose tip, right and left mouth corner."""

    confidence: float
    """Detector score. Not a recognition score — see :class:`Match`."""

    @property
    def area(self) -> int:
        """Pixel area, used to pick the most prominent face in a frame."""
        return self.bbox[2] * self.bbox[3]


@dataclass(frozen=True)
class Match:
    """A gallery entry that resembles a query embedding."""

    label: str

    score: float
    """Cosine similarity in ``[-1, 1]``. Higher is more alike."""

    entry_id: str
    """Opaque id of the stored entry, as returned by ``FaceStore.add``."""

    metadata: Mapping[str, Any]


@dataclass(frozen=True, eq=False)
class Recognition:
    """One detected face and the best gallery entry for it, if any cleared
    the threshold. ``match is None`` means unknown — the library does not
    invent a label for it."""

    face: Face
    match: Match | None


@dataclass(frozen=True, eq=False)
class StoreEntry:
    """One stored gallery entry, as handed out by a store that can enumerate.

    ``eq=False`` for the same reason as :class:`Face`: ``vector`` is an ndarray
    and a generated ``__eq__`` would return an array.

    Not part of the :class:`~alchemyface.store.base.FaceStore` protocol —
    searching is the protocol's job. Enumeration is what a user interface needs
    in order to list a database, so the stores that support it expose this.
    """

    entry_id: str
    label: str
    group: str
    vector: NDArray[np.float32]
    """As stored. :class:`~alchemyface.store.pickle.PickleStore` keeps SFace's
    raw magnitude here, so its L2 norm is meaningful rather than always 1."""
