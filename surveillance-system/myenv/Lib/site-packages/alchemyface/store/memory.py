"""A gallery held in a single numpy matrix.

Every vector is stored L2-normalised, which makes cosine similarity a matrix
product: ``vectors @ query``. That is why scikit-learn is not a dependency.
Brute force over a few thousand faces is well under a millisecond, and it
keeps the default install free of any database.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from alchemyface.types import Match


@dataclass(frozen=True)
class _Entry:
    entry_id: str
    label: str
    metadata: dict[str, Any]


def _as_unit_row(vector: NDArray[np.float32], dim: int) -> NDArray[np.float32]:
    """Validate, flatten and normalise. Accepts ``(dim,)`` or ``(1, dim)``."""
    flat = np.asarray(vector, dtype=np.float32).ravel()
    if flat.shape != (dim,):
        raise ValueError(f"expected a vector of dimension {dim}, got shape {np.shape(vector)}")
    norm = float(np.linalg.norm(flat))
    if norm == 0.0:
        raise ValueError("cannot store a zero vector: it has no direction")
    return (flat / norm).astype(np.float32)


class InMemoryStore:
    """Implements :class:`~alchemyface.store.base.FaceStore` with numpy."""

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim
        self._vectors: NDArray[np.float32] = np.empty((0, dim), dtype=np.float32)
        self._entries: list[_Entry] = []

    @property
    def dim(self) -> int:
        """Length of the vectors this gallery holds."""
        return self._dim

    def __len__(self) -> int:
        """How many entries are stored."""
        return len(self._entries)

    def add(
        self,
        label: str,
        vector: NDArray[np.float32],
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Store an embedding under a label. Returns its opaque entry id."""
        row = _as_unit_row(vector, self._dim)
        entry = _Entry(uuid4().hex, label, dict(metadata or {}))
        self._vectors = np.vstack([self._vectors, row])
        self._entries.append(entry)
        return entry.entry_id

    def search(self, vector: NDArray[np.float32], k: int = 1) -> list[Match]:
        """The ``k`` most similar entries, best first."""
        if not self._entries:
            return []
        query = _as_unit_row(vector, self._dim)
        scores = self._vectors @ query
        take = min(max(k, 1), len(self._entries))
        # argpartition finds the top-k in O(n); only those k are then sorted.
        top = np.argpartition(-scores, take - 1)[:take]
        top = top[np.argsort(-scores[top])]
        return [
            Match(
                label=self._entries[i].label,
                score=float(scores[i]),
                entry_id=self._entries[i].entry_id,
                metadata=dict(self._entries[i].metadata),
            )
            for i in top
        ]

    def remove(self, entry_id: str) -> None:
        """Delete one entry. Raises ``KeyError`` if it is not there."""
        for index, entry in enumerate(self._entries):
            if entry.entry_id == entry_id:
                del self._entries[index]
                self._vectors = np.delete(self._vectors, index, axis=0)
                return
        raise KeyError(entry_id)

    def save(self, path: Path | str) -> None:
        """Write the gallery to a ``.npz`` file.

        Metadata goes in as one JSON blob rather than an object array, so the
        file loads without ``allow_pickle`` and cannot execute anything.
        """
        np.savez(
            path,
            vectors=self._vectors,
            labels=np.array([e.label for e in self._entries], dtype="U"),
            entry_ids=np.array([e.entry_id for e in self._entries], dtype="U"),
            metadata=np.array(json.dumps([e.metadata for e in self._entries])),
            dim=np.array(self._dim),
        )

    def load(self, path: Path | str) -> None:
        """Replace the gallery with the contents of a ``.npz`` file."""
        with np.load(path, allow_pickle=False) as data:
            dim = int(data["dim"])
            if dim != self._dim:
                raise ValueError(f"gallery dimension {dim} does not match store dimension {self._dim}")
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            labels: list[str] = np.asarray(data["labels"]).tolist()
            entry_ids: list[str] = np.asarray(data["entry_ids"]).tolist()
            metadata = json.loads(str(data["metadata"]))

        self._vectors = vectors.reshape(-1, self._dim)
        self._entries = [
            _Entry(entry_id, label, dict(meta)) for entry_id, label, meta in zip(entry_ids, labels, metadata)
        ]
