"""The detection seam."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from alchemyface.types import Face


@runtime_checkable
class Detector(Protocol):
    """Finds faces in a BGR image."""

    def detect(self, image: NDArray[np.uint8]) -> list[Face]:
        """Every face found, in the detector's own order. Empty list if none."""
