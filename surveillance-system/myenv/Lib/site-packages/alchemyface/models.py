"""Locating the ONNX weights AlchemyFace runs on.

Weights are not vendored: a 37 MB wheel is slow to install and re-uploads on
every release. They are resolved at runtime, first hit wins:

1. an explicit ``model_dir`` argument
2. ``$ALCHEMYFACE_MODEL_DIR``
3. ``~/.cache/alchemyface/models/`` (honours ``$XDG_CACHE_HOME``)
4. downloaded from the OpenCV Zoo and verified against a pinned SHA256

Files already on disk are trusted and not checksummed — they may legitimately
be a different build. Downloads always are.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from alchemyface.errors import ModelDownloadError, ModelNotFoundError

# The zoo keeps its weights in Git LFS. raw.githubusercontent.com serves the
# ~130-byte pointer file instead of the model; the media host serves the model.
_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"

_CHUNK = 1 << 20


@dataclass(frozen=True)
class ModelSpec:
    """Where a model lives, what it is called, and how to know it arrived intact."""

    key: str
    filename: str
    url: str
    sha256: str

    aliases: tuple[str, ...] = ()
    """Other filenames the same architecture ships under. The prototype's
    weights use different names, and should resolve without being renamed."""

    @property
    def candidates(self) -> tuple[str, ...]:
        """Filenames to look for on disk, canonical name first."""
        return (self.filename, *self.aliases)


DETECTOR = ModelSpec(
    key="detector",
    filename="face_detection_yunet_2023mar.onnx",
    url=f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    aliases=("yunet_n_640_640.onnx",),
)

EMBEDDER = ModelSpec(
    key="embedder",
    filename="face_recognition_sface_2021dec.onnx",
    url=f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    aliases=("face_recognizer_fast.onnx",),
)

MODELS: dict[str, ModelSpec] = {spec.key: spec for spec in (DETECTOR, EMBEDDER)}


def cache_dir() -> Path:
    """Where downloaded weights are kept between runs."""
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "alchemyface" / "models"


def _search_paths(model_dir: Path | str | None = None) -> Iterator[Path]:
    if model_dir is not None:
        yield Path(model_dir)
    env = os.environ.get("ALCHEMYFACE_MODEL_DIR")
    if env:
        yield Path(env)
    yield cache_dir()


def find_local(spec: ModelSpec, model_dir: Path | str | None = None) -> Path | None:
    """First existing file matching any of the spec's names, or None."""
    for directory in _search_paths(model_dir):
        for name in spec.candidates:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def find_in(spec: ModelSpec, directory: Path | str) -> Path | None:
    """First existing file matching the spec's names in *exactly* this directory.

    Unlike :func:`find_local` this does not fall back to the environment
    variable or the cache. A caller that named a directory wants to know about
    that directory: answering "already present" because the file is somewhere
    else entirely is not useful.
    """
    for name in spec.candidates:
        candidate = Path(directory) / name
        if candidate.is_file():
            return candidate
    return None


def sha256_of(path: Path | str) -> str:
    """Hex digest of a file, read in chunks so a 37 MB model is not slurped."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(spec: ModelSpec, dest_dir: Path | str | None = None) -> Path:
    """Fetch a model, verify it, and place it atomically.

    The download goes to a ``.part`` file in the destination directory and is
    renamed only after the checksum passes, so an interrupted or corrupted
    download can never be mistaken for a usable cache entry.
    """
    directory = Path(dest_dir) if dest_dir is not None else cache_dir()
    # Wrapped: a caller can name an unwritable directory, and every other model
    # failure in this module surfaces as an AlchemyFaceError. Letting a raw
    # PermissionError through would make the destination the one case a caller
    # had to catch OSError for.
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handle, raw_tmp = tempfile.mkstemp(dir=directory, suffix=".part")
    except OSError as exc:
        raise ModelDownloadError(f"cannot write to {directory}: {exc}") from exc
    os.close(handle)
    tmp = Path(raw_tmp)

    try:
        with urllib.request.urlopen(spec.url, timeout=60) as response:
            with open(tmp, "wb") as out:
                shutil.copyfileobj(response, out, _CHUNK)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        tmp.unlink(missing_ok=True)
        raise ModelDownloadError(f"could not download {spec.filename} from {spec.url}: {exc}") from exc

    actual = sha256_of(tmp)
    if actual != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise ModelDownloadError(f"checksum mismatch for {spec.filename}: expected {spec.sha256}, got {actual}")

    destination = directory / spec.filename
    tmp.replace(destination)
    return destination


def resolve(
    spec: ModelSpec,
    model_dir: Path | str | None = None,
    allow_download: bool = True,
) -> Path:
    """Path to usable weights, downloading them if that is permitted."""
    local = find_local(spec, model_dir)
    if local is not None:
        return local
    if not allow_download:
        raise ModelNotFoundError(
            f"{spec.filename} was not found in any of "
            f"{[str(p) for p in _search_paths(model_dir)]}, "
            "and downloading was disabled. Set ALCHEMYFACE_MODEL_DIR, pass "
            "model_dir=, or run `alchemyface download-models`."
        )
    # Into ``model_dir`` when one was named, not the cache. Dropping it here
    # meant a caller who asked for the weights in a particular place got them
    # silently written somewhere else.
    return download(spec, dest_dir=model_dir)
