"""What the Inspect tab displays, computed without any reference to Tk.

Kept apart from the widget so it can be imported and tested anywhere — the
widget module imports ``tkinter``, which is not installed everywhere Python is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alchemyface.store import PickleStore
from alchemyface.types import StoreEntry

PREVIEW_VALUES = 5
"""How many leading vector values the table previews."""

BLANK_GROUP = "—"
"""Shown instead of an empty group, so the column never looks broken."""


@dataclass(frozen=True)
class EntryRow:
    """One row of the entries table, ready to display."""

    index: int
    """1-based, for display."""

    entry_id: str
    name: str
    group: str
    dim: int
    norm: float
    preview: str


@dataclass(frozen=True)
class DatabaseSummary:
    """The one-line summary above the table."""

    entries: int
    unique_names: int
    groups: int
    dim: int
    size_bytes: int
    path: Path | None

    def __str__(self) -> str:
        dim = self.dim if self.dim else BLANK_GROUP
        text = (
            f"{self.entries} entries · {self.unique_names} unique names · "
            f"{self.groups} groups · dim={dim} · {self.size_bytes / 1024:.1f} KB"
        )
        return f"{text} · {self.path}" if self.path is not None else text


def _preview(vector: np.ndarray) -> str:
    head = vector[:PREVIEW_VALUES].tolist()
    body = ", ".join(f"{value:+.4f}" for value in head)
    tail = ", …]" if vector.size > PREVIEW_VALUES else "]"
    return f"[{body}{tail}"


def entry_rows(store: PickleStore) -> list[EntryRow]:
    """Table rows for every entry in the store."""
    return [
        EntryRow(
            index=index,
            entry_id=entry.entry_id,
            name=entry.label,
            group=entry.group or BLANK_GROUP,
            dim=int(entry.vector.size),
            norm=float(np.linalg.norm(entry.vector)),
            preview=_preview(entry.vector),
        )
        for index, entry in enumerate(store.entries(), start=1)
    ]


def summarise(store: PickleStore, path: Path | None) -> DatabaseSummary:
    """Counts and file size for the summary line."""
    entries: list[StoreEntry] = store.entries()
    size = 0
    if path is not None:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
    return DatabaseSummary(
        entries=len(entries),
        unique_names=len({e.label for e in entries}),
        groups=len({e.group for e in entries if e.group}),
        dim=int(entries[0].vector.size) if entries else 0,
        size_bytes=size,
        path=path,
    )
