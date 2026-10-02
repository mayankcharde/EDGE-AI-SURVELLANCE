"""The gallery seam.

A store owns enrolled embeddings and answers nearest-neighbour queries. It is
a Protocol rather than a base class so a pgvector or SQLite implementation can
be added later without inheriting from anything in this package.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from alchemyface.types import Match


@runtime_checkable
class FaceStore(Protocol):
    """Holds labelled embeddings and finds the closest ones."""

    def add(
        self,
        label: str,
        vector: NDArray[np.float32],
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Store an embedding under a label. Returns its opaque entry id."""

    def search(self, vector: NDArray[np.float32], k: int = 1) -> list[Match]:
        """The ``k`` most similar entries, best first. Empty if the store is."""

    def remove(self, entry_id: str) -> None:
        """Delete one entry. Raises ``KeyError`` if it is not there."""

    def __len__(self) -> int:
        """How many entries are stored."""
