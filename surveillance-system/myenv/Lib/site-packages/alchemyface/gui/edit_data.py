"""The Edit DB tab's session model, with no reference to Tk.

Editing an existing database is a small state machine: entries loaded from a
file, additions waiting to be merged, and whether anything has changed since the
last save. The original kept all three tangled into widget callbacks, so none of
it could be tested. Here it is a class.

Additions are held apart from entries deliberately. A face detected from a photo
is a *candidate* — the user names it, picks a group, and may untick it — and
nothing about the database has changed until they merge. Which is why adding
pending faces does not mark the session dirty, and merging does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from alchemyface.gui.annotation_data import DEFAULT_GROUP
from alchemyface.gui.inspect_data import BLANK_GROUP, PREVIEW_VALUES
from alchemyface.store import PickleStore
from alchemyface.store.pickle import DEFAULT_DIM
from alchemyface.types import Face


@dataclass
class EditableEntry:
    """One row of the database being edited."""

    name: str
    group: str
    vector: NDArray[np.float32]
    """As loaded — raw, unnormalised, unchanged by a round trip."""


@dataclass
class PendingFace:
    """A face detected from a photo, not yet part of the database."""

    source: Path
    face: Face
    include: bool = True
    name: str = ""
    group: str = DEFAULT_GROUP
    embedding: NDArray[np.float32] | None = field(default=None, repr=False)


@dataclass(frozen=True)
class MergeResult:
    """What ``merge_checked`` managed to do.

    ``skipped`` names each candidate that could not be added and why. Those
    stay pending, so the user can fix them and try again.
    """

    added: int
    skipped: list[str]


@dataclass(frozen=True)
class EditRow:
    """One row of the entries table, ready to display."""

    index: int
    """1-based, for display."""

    name: str
    group: str
    dim: int
    norm: float
    preview: str


class EditSession:
    """Entries, pending additions, and whether anything needs saving."""

    def __init__(self) -> None:
        self._entries: list[EditableEntry] = []
        self._pending: list[PendingFace] = []
        self._problems: list[str] = []
        self._path: Path | None = None
        self._dirty = False

    # ---------------------------------------------------------------- state

    @property
    def entries(self) -> list[EditableEntry]:
        return self._entries

    @property
    def pending(self) -> list[PendingFace]:
        return self._pending

    @property
    def path(self) -> Path | None:
        """Where this was loaded from, or last saved to."""
        return self._path

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def dim(self) -> int:
        """Vector dimension of what is loaded, or the SFace default if empty."""
        return int(self._entries[0].vector.size) if self._entries else DEFAULT_DIM

    @property
    def title_suffix(self) -> str:
        """`" *"` while there are unsaved changes, so the frame title shows it."""
        return " *" if self._dirty else ""

    def groups_in_use(self) -> set[str]:
        """Every non-blank group currently present, to offer as presets."""
        return {entry.group for entry in self._entries if entry.group}

    # --------------------------------------------------------------- loading

    def load(self, path: Path | str) -> None:
        """Replace everything with the contents of a database file.

        Pending additions are discarded: they were candidates for the previous
        database and mean nothing against a different one.
        """
        store = PickleStore()
        # Lenient so a database holding a bad vector can still be opened and
        # repaired here — deleting the offending row is the whole reason to
        # bring it into the editor. Saving stays strict, so a row that cannot
        # be matched against is never written back out.
        self._problems = store.load_leniently(path)
        self._entries = [
            EditableEntry(name=entry.label, group=entry.group, vector=entry.vector) for entry in store.entries()
        ]
        self._pending = []
        self._path = Path(path)
        self._dirty = False

    @property
    def problems(self) -> list[str]:
        """What was wrong with the loaded file, if anything."""
        return list(self._problems)

    # --------------------------------------------------------------- editing

    def remove(self, indices: list[int] | set[int]) -> int:
        """Drop rows by index. Returns how many went.

        Indices are removed high-to-low, because deleting ascending would shift
        the later ones out from under the loop.
        """
        wanted = sorted({i for i in indices if 0 <= i < len(self._entries)}, reverse=True)
        for index in wanted:
            del self._entries[index]
        if wanted:
            self._dirty = True
        return len(wanted)

    def set_group(self, index: int, group: str) -> None:
        """Change one row's group. Blank is allowed — the schema permits it."""
        if not 0 <= index < len(self._entries):
            return
        value = group.strip()
        if self._entries[index].group == value:
            return
        self._entries[index].group = value
        self._dirty = True

    def set_name(self, index: int, name: str) -> None:
        if not 0 <= index < len(self._entries):
            return
        value = name.strip()
        if not value or self._entries[index].name == value:
            return
        self._entries[index].name = value
        self._dirty = True

    # ------------------------------------------------------------- additions

    def add_pending(self, faces: list[PendingFace]) -> None:
        """Offer newly detected faces as candidates. Not a change yet."""
        self._pending.extend(faces)

    def clear_pending(self) -> None:
        self._pending = []

    def merge_checked(self) -> MergeResult:
        """Move every usable ticked candidate into the entries table.

        A candidate with no name or no embedding cannot be matched against, so
        it is skipped rather than written — but only *it* is skipped, and the
        rest of the batch still lands. Refusing the whole batch over one blank
        name meant re-ticking a screenful of faces to fix a single typo.

        Nothing is discarded: a candidate that was skipped, and one that was
        never ticked, both stay pending, so they can be named, ticked and added
        on a second pass. Duplicate names are fine — the robot resolves by best
        cosine similarity, so a second photo of someone is an improvement.
        """
        added: list[EditableEntry] = []
        skipped: list[str] = []
        keep: list[PendingFace] = []
        for face in self._pending:
            if not face.include:
                keep.append(face)
                continue
            if not face.name.strip():
                skipped.append(f"{face.source.name}: a face has no name")
                keep.append(face)
                continue
            if face.embedding is None:
                skipped.append(f"{face.source.name}: {face.name} has no embedding")
                keep.append(face)
                continue
            added.append(
                EditableEntry(
                    name=face.name.strip(),
                    group=face.group.strip(),
                    vector=face.embedding,
                )
            )
        self._entries.extend(added)
        self._pending = keep
        if added:
            self._dirty = True
        return MergeResult(added=len(added), skipped=skipped)

    # ---------------------------------------------------------------- saving

    def save(self, path: Path | str | None = None) -> Path:
        """Write the database. With no argument, writes where it came from."""
        destination = Path(path) if path is not None else self._path
        if destination is None:
            raise ValueError("no path to save to; pass one explicitly")
        # An empty table is almost always a mistake — a select-all-delete, or a
        # load that failed — and writing it would replace a real database with
        # an empty one, losing every entry, with no undo.
        if not self._entries:
            raise ValueError("Nothing to save — the entries list is empty.")
        # The store validates every vector against its dimension, so it must be
        # built for what this session actually holds. Defaulting to 128 worked
        # only because every real database happens to be 128.
        store = PickleStore(dim=self.dim)
        for entry in self._entries:
            store.add(entry.name, entry.vector, {"group": entry.group})
        store.save(destination)
        self._path = destination
        self._dirty = False
        return destination

    # ------------------------------------------------------------------ rows

    def rows(self) -> list[EditRow]:
        """Table rows for every entry."""
        return [
            EditRow(
                index=index,
                name=entry.name,
                group=entry.group or BLANK_GROUP,
                dim=int(entry.vector.size),
                norm=float(np.linalg.norm(entry.vector)),
                preview=_preview(entry.vector),
            )
            for index, entry in enumerate(self._entries, start=1)
        ]


def _preview(vector: NDArray[np.float32]) -> str:
    head = vector[:PREVIEW_VALUES].tolist()
    body = ", ".join(f"{value:+.4f}" for value in head)
    tail = ", …]" if vector.size > PREVIEW_VALUES else "]"
    return f"[{body}{tail}"
