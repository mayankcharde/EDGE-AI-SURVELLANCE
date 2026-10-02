"""A gallery backed by the robot's pickle format.

The Unitree G1's ``face_recognition_controller`` reads a pickle holding::

    [
        ("0", "Alice", "staff",   np.ndarray(shape=(128,), dtype=float32)),
        ("1", "Bob",   "visitor", np.ndarray(shape=(128,), dtype=float32)),
    ]

Two things distinguish this store from :class:`~alchemyface.store.memory.InMemoryStore`,
and both are deliberate.

**Vectors are stored exactly as given.** The robot's databases hold raw SFace
output, whose L2 norm is around 10, and a round trip through this store must not
alter a single value. ``search`` normalises internally instead, so matching is
unaffected — cosine similarity is scale-invariant.

**Reading is forgiving.** The production databases disagree with their own
documented schema: ``id`` is an ``int`` in one file and a ``str`` in others, and
the vector is ``(1, 128)`` in one and ``(128,)`` in others. Both are coerced. A
stricter reader would refuse a database the robot loads today.

Forgiving is not the same as permissive. Zero vectors are refused by both
:meth:`add` and :meth:`load`, unlike the original app's reader, because a zero
has no direction: ``search`` would divide by its norm and return ``NaN`` for
every query. Rejecting at the boundary is the only place that check works.

``search`` rebuilds its matrix on each call. With a few hundred entries — the
size of every real database here — that is far cheaper than maintaining an
incremental copy, and it keeps ``add`` and ``remove`` trivial.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from alchemyface.errors import PickleSchemaError
from alchemyface.types import Match, StoreEntry

DEFAULT_DIM = 128
"""Dimension of an SFace embedding, and of every known robot database."""


@dataclass
class _Entry:
    entry_id: str
    label: str
    group: str
    vector: NDArray[np.float32]


def _coerce_vector(value: object, dim: int | None, where: str) -> NDArray[np.float32]:
    """Flatten and validate one vector, or raise :class:`PickleSchemaError`.

    ``dim=None`` accepts whatever dimension the value has, which is how the
    first entry of a file establishes the dimension for the rest.
    """
    try:
        flat = np.asarray(value, dtype=np.float32).ravel()
    except (TypeError, ValueError) as exc:
        raise PickleSchemaError(f"{where}: not a numeric vector: {exc}") from exc
    # ravel() always yields one dimension, so only emptiness is reachable here.
    if flat.size == 0:
        raise PickleSchemaError(f"{where}: vector is empty")
    if dim is not None and flat.size != dim:
        raise PickleSchemaError(f"{where}: vector dimension {flat.size} does not match {dim}")
    if not np.isfinite(flat).all():
        raise PickleSchemaError(f"{where}: vector contains NaN or infinity")
    if float(np.linalg.norm(flat)) == 0.0:
        raise PickleSchemaError(f"{where}: zero vector has no direction")
    return flat


class PickleStore:
    """Implements :class:`~alchemyface.store.base.FaceStore` over the robot's pickle."""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self._dim = dim
        self._entries: list[_Entry] = []

    @property
    def dim(self) -> int:
        """Vector dimension. Adopted from a file on :meth:`load`."""
        return self._dim

    def __len__(self) -> int:
        return len(self._entries)

    def vectors(self) -> list[NDArray[np.float32]]:
        """Copies of the stored vectors, exactly as held — unnormalised.

        Copies, not references: a caller mutating a returned array would
        otherwise corrupt the gallery silently, and a zero written in that way
        would make :meth:`search` return ``NaN``.
        """
        return [entry.vector.copy() for entry in self._entries]

    def entries(self) -> list[StoreEntry]:
        """Every entry, in insertion order, with copied vectors.

        ``search`` answers "who is this?"; this answers "what is in here?",
        which is what a user interface listing a database needs.
        """
        return [
            StoreEntry(
                entry_id=entry.entry_id,
                label=entry.label,
                group=entry.group,
                vector=entry.vector.copy(),
            )
            for entry in self._entries
        ]

    def add(
        self,
        label: str,
        vector: NDArray[np.float32],
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Store a vector verbatim under a label. ``metadata["group"]`` is kept."""
        vec = _coerce_vector(vector, self._dim, f"entry {label!r}")
        group = str((metadata or {}).get("group", ""))
        entry = _Entry(uuid4().hex, str(label), group, vec)
        self._entries.append(entry)
        return entry.entry_id

    def search(self, vector: NDArray[np.float32], k: int = 1) -> list[Match]:
        """The ``k`` most similar entries, best first.

        Both sides are normalised here rather than on the way in, so stored
        magnitudes are preserved without affecting the ranking.
        """
        if not self._entries:
            return []
        query = _coerce_vector(vector, self._dim, "query")
        query = query / np.linalg.norm(query)

        matrix = np.vstack([entry.vector for entry in self._entries])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        scores = (matrix / norms) @ query

        take = min(max(k, 1), len(self._entries))
        top = np.argpartition(-scores, take - 1)[:take]
        top = top[np.argsort(-scores[top])]
        return [
            Match(
                label=self._entries[i].label,
                score=float(scores[i]),
                entry_id=self._entries[i].entry_id,
                metadata={"group": self._entries[i].group},
            )
            for i in top
        ]

    def remove(self, entry_id: str) -> None:
        """Delete one entry. Raises ``KeyError`` if it is not there."""
        for index, entry in enumerate(self._entries):
            if entry.entry_id == entry_id:
                del self._entries[index]
                return
        raise KeyError(entry_id)

    def save(self, path: Path | str) -> None:
        """Write the gallery in the robot's format, renumbering ids from ``"0"``."""
        database = [
            (str(i), entry.label, entry.group, np.asarray(entry.vector, dtype=np.float32))
            for i, entry in enumerate(self._entries)
        ]
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "wb") as handle:
            pickle.dump(database, handle)

    def load(self, path: Path | str) -> None:
        """Replace the gallery with the contents of a pickle.

        Accepts the four-tuple list form and the back-compatible
        ``{name: vector}`` dict. Anything else raises
        :class:`PickleSchemaError` with a reason.
        """
        data = self._unpickle(path)
        self._entries = self._parse(data)
        if self._entries:
            self._dim = int(self._entries[0].vector.size)

    @staticmethod
    def _unpickle(path: Path | str) -> object:
        try:
            with open(path, "rb") as handle:
                return pickle.load(handle)
        except (OSError, PermissionError) as exc:
            raise PickleSchemaError(f"{path}: cannot be read: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            # Unpickling can raise almost anything: UnpicklingError, EOFError,
            # ValueError, IndexError, AttributeError, and ModuleNotFoundError
            # when a GLOBAL opcode names a module that is not installed.
            # Enumerating them left real files crashing past the wrapper.
            # OSError is caught above so it keeps its own clearer message.
            raise PickleSchemaError(f"{path}: not a readable pickle: {exc}") from exc

    def load_leniently(self, path: Path | str) -> list[str]:
        """Read a database for inspection, keeping entries a strict load rejects.

        Returns a list of complaints, one per salvaged entry, and leaves the
        gallery holding everything it could parse — including vectors carrying
        NaN, infinity, or a dimension that disagrees with the rest.

        Refusing those outright was backwards for the two tools that use this.
        A database is opened in the inspector *because* something is wrong with
        it, and in the editor to delete the offending rows; aborting the load
        meant a file with one bad vector could be neither diagnosed nor
        repaired. :meth:`load` stays strict, so nothing enrols against a vector
        that cannot be matched.
        """
        data = self._unpickle(path)
        entries, problems = self._parse_leniently(data)
        self._entries = entries
        if entries:
            self._dim = int(entries[0].vector.size)
        return problems

    def _parse_leniently(self, data: object) -> tuple[list[_Entry], list[str]]:
        if isinstance(data, dict):
            items: list[tuple[object, object, object, object]] = [
                (index, name, "", vector) for index, (name, vector) in enumerate(data.items())
            ]
        elif isinstance(data, list):
            items = []
            problems: list[str] = []
            for index, item in enumerate(data):
                if isinstance(item, tuple) and len(item) == 4:
                    items.append(item)
                else:
                    problems.append(f"entry {index}: not a 4-tuple, skipped")
        else:
            raise PickleSchemaError(
                f"expected a list of (id, name, group, vector) tuples or a "
                f"{{name: vector}} dict, got {type(data).__name__}"
            )

        problems = problems if isinstance(data, list) else []
        entries: list[_Entry] = []
        for index, (entry_id, name, group, vector) in enumerate(items):
            try:
                flat = np.asarray(vector, dtype=np.float32).ravel()
            except (TypeError, ValueError) as exc:
                problems.append(f"entry {index} ({name!r}): not a numeric vector, skipped: {exc}")
                continue
            if flat.size == 0:
                problems.append(f"entry {index} ({name!r}): vector is empty, skipped")
                continue
            if not np.isfinite(flat).all():
                problems.append(f"entry {index} ({name!r}): vector contains NaN or infinity")
            entries.append(_Entry(str(entry_id), str(name), str(group), flat))

        if entries:
            first = entries[0].vector.size
            for index, entry in enumerate(entries):
                if entry.vector.size != first:
                    problems.append(f"entry {index} ({entry.label!r}): dimension {entry.vector.size}, expected {first}")
        return entries, problems

    def _parse(self, data: object) -> list[_Entry]:
        if isinstance(data, dict):
            return self._parse_dict(data)
        if isinstance(data, list):
            return self._parse_list(data)
        raise PickleSchemaError(
            f"expected a list of (id, name, group, vector) tuples or a {{name: vector}} dict, got {type(data).__name__}"
        )

    def _parse_dict(self, data: Mapping[object, object]) -> list[_Entry]:
        entries: list[_Entry] = []
        dim: int | None = None
        for index, (name, vector) in enumerate(data.items()):
            vec = _coerce_vector(vector, dim, f"entry {index} ({name!r})")
            dim = int(vec.size)
            # The index, not a fresh uuid: the dict form carries no ids, and a
            # random one differed on every load, so the ID column could not be
            # used to refer to a row or to compare two views of the same file.
            entries.append(_Entry(str(index), str(name), "", vec))
        return entries

    def _parse_list(self, data: list[object]) -> list[_Entry]:
        entries: list[_Entry] = []
        dim: int | None = None
        for index, item in enumerate(data):
            # The schema says tuples. A list here means something else wrote
            # the file, so it is not silently accepted.
            if not isinstance(item, tuple) or len(item) != 4:
                raise PickleSchemaError(
                    f"entry {index}: expected a 4-tuple "
                    f"(id, name, group, vector), got "
                    f"{type(item).__name__}" + (f" of length {len(item)}" if isinstance(item, (tuple, list)) else "")
                )
            entry_id, name, group, vector = item
            vec = _coerce_vector(vector, dim, f"entry {index} ({name!r})")
            dim = int(vec.size)
            entries.append(_Entry(str(entry_id), str(name), str(group), vec))
        return entries
