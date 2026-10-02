"""Bulk image resizing, with no reference to Tk.

This exists because of a real detector limitation: YuNet's largest anchors miss a
face that fills most of the frame, which is exactly what a phone selfie looks
like. Shrinking the photo brings the face back into range. Everything here is
ordinary file work, so the whole feature is testable without a display.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from alchemyface.gui.annotation_data import IMAGE_EXTENSIONS

MIN_RATIO = 0.05
MAX_RATIO = 5.0
DEFAULT_RATIO = 0.5
"""Half size. The common case is shrinking a selfie until YuNet can see it."""

JPEG_SAVE = {"quality": 95, "subsampling": 0}
WEBP_SAVE = {"quality": 95, "method": 6}
"""Deliberately near-lossless: these images are about to be enrolled, and
compression artefacts move the embedding."""


@dataclass(frozen=True)
class ResizeResult:
    """The dimensions before and after."""

    old_width: int
    old_height: int
    new_width: int
    new_height: int

    def __str__(self) -> str:
        return f"{self.old_width}x{self.old_height} -> {self.new_width}x{self.new_height}"


@dataclass(frozen=True)
class ResizeOutcome:
    """What happened to one file. A failure is data, not an exception."""

    source: Path
    destination: Path
    result: ResizeResult | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def __str__(self) -> str:
        if self.ok:
            return f"{self.source.name}: {self.result}"
        return f"{self.source.name}: FAILED ({self.error})"


def default_output_folder(source: Path | str) -> Path:
    """``photos`` becomes ``photos_resized``, beside the original."""
    path = Path(source)
    return path.with_name(f"{path.name}_resized")


def resize_one(source: Path | str, destination: Path | str, ratio: float) -> ResizeResult:
    """Resize one image and write it, returning the dimensions either side.

    Refuses to write over its own source: a resize is not reversible, so doing
    so would destroy the original with no way back.

    Uses LANCZOS when shrinking and BICUBIC when growing, applies the EXIF
    orientation so the saved file is upright, and saves near-losslessly for
    formats that would otherwise re-compress.
    """
    src, dst = Path(source), Path(destination)
    if src.resolve() == dst.resolve() if src.exists() else src == dst:
        raise ValueError(f"source and destination are the same file: {src}")

    with Image.open(src) as opened:
        opened.load()
        image = ImageOps.exif_transpose(opened) or opened
        old_width, old_height = image.width, image.height

        new_width = max(1, int(round(old_width * ratio)))
        new_height = max(1, int(round(old_height * ratio)))
        resampling = Image.Resampling.LANCZOS if ratio < 1.0 else Image.Resampling.BICUBIC
        resized = image.resize((new_width, new_height), resampling)

        # Any, not object: PIL's save() takes **params after a `format`
        # positional, so a dict[str, object] splat is matched against `format`.
        options: dict[str, Any] = {}
        suffix = dst.suffix.lower() or src.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            if resized.mode not in ("RGB", "L"):
                resized = resized.convert("RGB")  # JPEG has no alpha channel
            options = dict(JPEG_SAVE)
        elif suffix == ".webp":
            options = dict(WEBP_SAVE)

        dst.parent.mkdir(parents=True, exist_ok=True)
        resized.save(dst, **options)

    return ResizeResult(old_width, old_height, new_width, new_height)


def plan_folder(source: Path | str, destination: Path | str, ratio: float) -> list[tuple[Path, Path]]:
    """Every (source, destination) pair a folder resize would write.

    Filenames are preserved, so the destination must differ from the source or
    every original would be overwritten.
    """
    src, dst = Path(source), Path(destination)
    if not src.is_dir():
        raise NotADirectoryError(f"not a folder: {src}")
    if src.resolve() == dst.resolve() if dst.exists() else src == dst:
        raise ValueError(f"source and destination are the same folder: {src}")
    del ratio  # planning does not depend on it; kept for a symmetric signature
    return [
        (path, dst / path.name)
        for path in sorted(src.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def resize_one_outcome(source: Path, destination: Path, ratio: float) -> ResizeOutcome:
    """One file's resize, with any failure captured rather than raised.

    Separate from :func:`resize_folder` so a caller can drive the loop itself
    and report each file as it lands — the GUI logs progress this way.
    """
    try:
        return ResizeOutcome(source, destination, result=resize_one(source, destination, ratio))
    except Exception as exc:  # noqa: BLE001 - a failure is one line of the log
        return ResizeOutcome(source, destination, error=f"{type(exc).__name__}: {exc}")


def resize_folder(source: Path | str, destination: Path | str, ratio: float) -> list[ResizeOutcome]:
    """Resize every image in a folder, reporting each file separately.

    One unreadable file must not abandon the rest of the batch, so failures are
    collected rather than raised.
    """
    return [resize_one_outcome(src, dst, ratio) for src, dst in plan_folder(source, destination, ratio)]
