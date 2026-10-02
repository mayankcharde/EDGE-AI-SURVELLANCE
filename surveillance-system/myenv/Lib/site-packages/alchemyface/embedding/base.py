"""The embedding seam."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from alchemyface.types import Face


@runtime_checkable
class Embedder(Protocol):
    """Turns a detected face into a comparable vector."""

    dim: int
    """Length of the vectors this embedder produces."""

    def embed(self, image: NDArray[np.uint8], face: Face) -> NDArray[np.float32]:
        """A unit-length embedding of ``face`` as it appears in ``image``.

        Implementations MUST return an L2-normalised, one-dimensional array of
        length ``dim``, so that a dot product between two of them is their
        cosine similarity.
        """
