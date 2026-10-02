"""The Build tab's data layer, with no reference to Tk.

The original app kept all of this inside its 816-line annotation widget, where
the canvas arithmetic and the sidebar state machine could only be exercised by
clicking. Here they are ordinary functions, so they carry ordinary tests and the
widget is left doing nothing but display.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from alchemyface.types import Face

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
"""What the Build tab will pick up from a folder."""

BOX_COLOURS = (
    "#ff3838",
    "#3aff3a",
    "#3a8bff",
    "#ffd13a",
    "#ff3aff",
    "#3affff",
    "#ff8c3a",
    "#9b3aff",
)
"""Cycled per face so numbered boxes on the canvas match the panel on the right."""

DEFAULT_GROUP = "staff"
BGR_CACHE_CAPACITY = 8
"""Decoded images are large; a handful is enough for navigation and prefetch."""


# ----------------------------------------------------------------- models


@dataclass
class FaceAnnotation:
    """One detected face plus the decisions a user has made about it."""

    face: Face
    include: bool = True
    name: str = ""
    group: str = DEFAULT_GROUP
    embedding: NDArray[np.float32] | None = field(default=None, repr=False)
    """Filled in on demand and kept. Excluding a face does not discard it."""

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.face.bbox


class EntryStatus(Enum):
    """Where an image has got to, which drives its sidebar row.

    ``FAILED`` is separate from ``NO_FACE`` on purpose. Reporting a failure as
    "no face detected" makes a broken model, an unreadable file or a missing
    recognizer look like a legitimate result about the photograph — which once
    led to a false claim in this project's own changelog about an image YuNet
    supposedly could not handle.
    """

    PENDING = "pending"
    FAILED = "detection failed"
    NO_FACE = "no face detected"
    NONE_INCLUDED = "detected, nothing included"
    INCLUDED = "included"


@dataclass
class ImageEntry:
    """One image in the folder, and what has been found in it."""

    path: Path
    detected: bool = False
    faces: list[FaceAnnotation] = field(default_factory=list)
    error: str | None = None
    """Why detection failed, if it did. Kept apart from an empty face list so a
    failure is never shown as a finding about the photograph."""

    edited: bool = False
    """Set once a user changes something, so Re-detect can warn before
    discarding their work."""

    @property
    def included_count(self) -> int:
        return sum(1 for face in self.faces if face.include)

    @property
    def status(self) -> EntryStatus:
        if not self.detected:
            return EntryStatus.PENDING
        if self.error is not None:
            return EntryStatus.FAILED
        if not self.faces:
            return EntryStatus.NO_FACE
        if self.included_count == 0:
            return EntryStatus.NONE_INCLUDED
        return EntryStatus.INCLUDED


# ------------------------------------------------------------ presentation

_GLYPHS = {
    EntryStatus.PENDING: "·",
    EntryStatus.FAILED: "✗",
    EntryStatus.NO_FACE: "⚠",
    EntryStatus.NONE_INCLUDED: "○",
    EntryStatus.INCLUDED: "✓",
}

_COLOURS = {
    EntryStatus.PENDING: "#888888",
    EntryStatus.FAILED: "#cc2200",
    EntryStatus.NO_FACE: "#cc7700",
    EntryStatus.NONE_INCLUDED: "#666666",
    EntryStatus.INCLUDED: "#1a7d1a",
}


def default_face_name(stem: str, index: int, total: int) -> str:
    """The name a face starts with: the filename, or filename_faceN if several."""
    return stem if total == 1 else f"{stem}_face{index + 1}"


def sidebar_text(entry: ImageEntry) -> str:
    """One sidebar row: a status glyph, the filename, and the included ratio."""
    glyph = _GLYPHS[entry.status]
    if entry.status is EntryStatus.FAILED:
        return f"{glyph}  {entry.path.name}  ({entry.error})"
    if entry.status in (EntryStatus.PENDING, EntryStatus.NO_FACE):
        return f"{glyph}  {entry.path.name}"
    return f"{glyph}  {entry.path.name}  ({entry.included_count}/{len(entry.faces)})"


def sidebar_colour(entry: ImageEntry) -> str:
    """Row foreground, so the list scans at a glance."""
    return _COLOURS[entry.status]


# -------------------------------------------------------- canvas geometry


@dataclass(frozen=True)
class FitTransform:
    """How an image sits on the canvas: scaled to fit, then centred."""

    scale: float
    offset_x: int
    offset_y: int
    width: int
    height: int

    def to_canvas(self, image_x: float, image_y: float) -> tuple[int, int]:
        return (
            self.offset_x + int(image_x * self.scale),
            self.offset_y + int(image_y * self.scale),
        )

    def to_image(self, canvas_x: float, canvas_y: float) -> tuple[float, float]:
        return (
            (canvas_x - self.offset_x) / self.scale,
            (canvas_y - self.offset_y) / self.scale,
        )


def fit_image(image_w: int, image_h: int, canvas_w: int, canvas_h: int) -> FitTransform:
    """Scale an image down to fit the canvas and centre it.

    Never scales up: enlarging a small photo to fill the canvas only blurs it.
    Tk reports a 1x1 canvas before its first layout pass, so the dimensions are
    floored at 1 rather than producing a zero or negative scale.
    """
    canvas_w = max(1, canvas_w)
    canvas_h = max(1, canvas_h)
    image_w = max(1, image_w)
    image_h = max(1, image_h)
    scale = min(canvas_w / image_w, canvas_h / image_h, 1.0)
    width = max(1, int(image_w * scale))
    height = max(1, int(image_h * scale))
    return FitTransform(
        scale=scale,
        offset_x=(canvas_w - width) // 2,
        offset_y=(canvas_h - height) // 2,
        width=width,
        height=height,
    )


def face_at(faces: list[FaceAnnotation], image_x: float, image_y: float) -> int | None:
    """Index of the first face whose box contains the point, or None.

    First rather than smallest or topmost: with overlapping detections the
    answer should be predictable — whichever the detector listed first.
    """
    for index, annotation in enumerate(faces):
        x, y, w, h = annotation.bbox
        if x <= image_x <= x + w and y <= image_y <= y + h:
            return index
    return None


# ------------------------------------------------------------------ saving


def records_for_save(entries: list[ImageEntry]) -> list[FaceAnnotation]:
    """Every included face across every detected image, in order."""
    return [annotation for entry in entries if entry.detected for annotation in entry.faces if annotation.include]


def validate_for_save(entries: list[ImageEntry]) -> list[str]:
    """Reasons the current state cannot be saved, one per offending face.

    Only naming is checked here. Whether an embedding exists is a question for
    whoever holds the model, not for this module.
    """
    errors: list[str] = []
    for entry in entries:
        if not entry.detected:
            continue
        for index, annotation in enumerate(entry.faces):
            if annotation.include and not annotation.name.strip():
                errors.append(f"{entry.path.name}: face #{index + 1} has no name.")
    return errors


# ------------------------------------------------------------- image cache


class BgrCache:
    """A small thread-safe LRU of decoded BGR images.

    Decoded images are megabytes each, so they are cached rather than attached
    to every entry. The worker thread and the Tk thread both reach for them,
    hence the lock; the disk read itself happens outside it, so readers do not
    serialise behind one another.
    """

    def __init__(self, capacity: int = BGR_CACHE_CAPACITY) -> None:
        self._capacity = max(1, capacity)
        self._items: OrderedDict[Path, NDArray[np.uint8]] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def get(self, path: Path) -> NDArray[np.uint8] | None:
        with self._lock:
            if path not in self._items:
                return None
            self._items.move_to_end(path)
            return self._items[path]

    def put(self, path: Path, image: NDArray[np.uint8]) -> None:
        with self._lock:
            self._items[path] = image
            self._items.move_to_end(path)
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)

    def drop(self, path: Path) -> None:
        with self._lock:
            self._items.pop(path, None)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
