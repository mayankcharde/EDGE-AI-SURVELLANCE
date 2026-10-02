"""Command line entry point for AlchemyFace."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from numpy.typing import NDArray

from alchemyface import __version__
from alchemyface.errors import AlchemyFaceError
from alchemyface.models import MODELS, download, find_in, find_local
from alchemyface.pipeline import DEFAULT_THRESHOLD, Recognizer
from alchemyface.store.memory import InMemoryStore

app = typer.Typer(
    name="alchemyface",
    help="Face detection and recognition on YuNet and SFace.",
    no_args_is_help=True,
    add_completion=False,
)


# Typer promotes a lone command to the top level, which would make `alchemyface
# version` an unexpected argument. An explicit callback keeps subcommand mode on
# regardless of how many commands are registered.
@app.callback()
def main() -> None:
    """Face detection and recognition on YuNet and SFace."""


def _build_recognizer(**kwargs: object) -> Recognizer:
    """Indirection so tests can substitute a fake-backed Recognizer."""
    return Recognizer(**kwargs)  # type: ignore[arg-type]


def _read_image(path: Path) -> NDArray[np.uint8]:
    # Imported lazily so `alchemyface version` does not pay for loading cv2.
    import cv2  # pylint: disable=import-outside-toplevel

    image = cv2.imread(str(path))
    if image is None:
        typer.echo(f"could not read an image from {path}", err=True)
        raise typer.Exit(code=2)
    # imread's default IMREAD_COLOR always yields 8-bit 3-channel BGR, so this
    # is a no-op at runtime; it is here to state the dtype for the type checker.
    return np.asarray(image, dtype=np.uint8)


def _load_gallery(recognizer: Recognizer, gallery: Path) -> None:
    store = recognizer.store
    if gallery.exists() and isinstance(store, InMemoryStore):
        store.load(gallery)


def _save_gallery(recognizer: Recognizer, gallery: Path) -> None:
    store = recognizer.store
    if isinstance(store, InMemoryStore):
        gallery.parent.mkdir(parents=True, exist_ok=True)
        store.save(gallery)


@app.command()
def version() -> None:
    """Print the installed AlchemyFace version."""
    typer.echo(__version__)


@app.command()
def db() -> None:
    """Launch the Face DB Builder desktop application."""
    # Imported here rather than at module scope on purpose. tkinter is a
    # separate OS package on Debian and Ubuntu, so importing it at the top would
    # make every `alchemyface` command fail — `--help` included — for anyone who
    # only wanted the library. (`import alchemyface` is unaffected either way:
    # the package __init__ never imports this module.)
    try:
        from alchemyface.gui.app import main as run_app  # noqa: PLC0415
    except ImportError as exc:
        typer.echo(
            f"the desktop application needs tkinter, which is not available: {exc}\n"
            "tkinter ships with Python but is packaged separately on some systems.\n"
            "  Debian/Ubuntu:  sudo apt-get install python3-tk\n"
            "  Fedora:         sudo dnf install python3-tkinter\n"
            "  macOS/Windows:  reinstall Python from python.org",
            err=True,
        )
        raise typer.Exit(code=3) from exc
    run_app()


@app.command()
def resize(
    ratio: float = typer.Option(0.5, "--ratio", help="Scale factor. 0.5 halves; 2.0 doubles."),
    folder: Path | None = typer.Option(None, "--folder", exists=True, file_okay=False, help="Resize every image here."),
    image: Path | None = typer.Option(None, "--image", exists=True, dir_okay=False, help="Resize a single image."),
    output: Path | None = typer.Option(None, "--output", help="Destination. Defaults beside the source, suffixed."),
) -> None:
    """Resize images so a too-close face comes back into the detector's range.

    YuNet's largest anchors miss a face filling most of the frame — a phone
    selfie held at arm's length. Shrinking the photo recovers it. Detection is
    not monotonic in the ratio, because it depends on the face matching an
    anchor scale, so if one value does not work another may.
    """
    # Imported here so the command costs nothing until used, and so this module
    # stays importable where Pillow is unavailable.
    from alchemyface.gui.resize_data import (  # noqa: PLC0415
        MAX_RATIO,
        MIN_RATIO,
        default_output_folder,
        resize_folder,
        resize_one,
    )

    if (folder is None) == (image is None):
        typer.echo("give exactly one of --folder or --image", err=True)
        raise typer.Exit(code=2)

    # Refused rather than clamped, matching the Resize tab. Silently turning a
    # mistyped 50 into 5.0 would rewrite the files at a size nobody asked for,
    # and a resize cannot be undone.
    if not MIN_RATIO <= ratio <= MAX_RATIO:
        typer.echo(f"--ratio must be between {MIN_RATIO} and {MAX_RATIO}, got {ratio:g}", err=True)
        raise typer.Exit(code=2)
    scale = ratio

    if folder is not None:
        destination = output or default_output_folder(folder)
        try:
            outcomes = resize_folder(folder, destination, scale)
        except (ValueError, NotADirectoryError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2) from exc
        if not outcomes:
            typer.echo(f"no images found in {folder}", err=True)
            raise typer.Exit(code=1)
        for outcome in outcomes:
            typer.echo(f"  {outcome}")
        written = sum(1 for o in outcomes if o.ok)
        typer.echo(f"{written} resized, {len(outcomes) - written} failed → {destination}")
        if written == 0:
            raise typer.Exit(code=1)
        return

    assert image is not None
    destination = output or image.with_name(f"{image.stem}_resized{image.suffix}")
    try:
        result = resize_one(image, destination, scale)
    except (OSError, ValueError) as exc:
        typer.echo(f"{image.name}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"  {image.name}: {result}")
    typer.echo(f"1 resized → {destination}")


@app.command("download-models")
def download_models(
    model_dir: Path | None = typer.Option(
        None,
        "--model-dir",
        help="Directory to write the weights into. Defaults to the cache.",
    ),
) -> None:
    """Fetch the ONNX weights ahead of first use."""
    for spec in MODELS.values():
        # With an explicit --model-dir, only that directory counts as present.
        # Reporting "already present" because a copy sits in the cache left the
        # requested directory empty, which is not what the flag promises.
        existing = find_in(spec, model_dir) if model_dir is not None else find_local(spec)
        if existing is not None:
            typer.echo(f"{spec.key}: already present at {existing}")
            continue
        typer.echo(f"{spec.key}: downloading {spec.filename} …")
        try:
            path = download(spec, dest_dir=model_dir)
        except AlchemyFaceError as exc:
            typer.echo(f"{spec.key}: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(f"{spec.key}: saved to {path}")


@app.command()
def enroll(
    name: str = typer.Option(..., "--name", help="Label to store the face under."),
    image: Path = typer.Option(..., "--image", exists=True, dir_okay=False, help="Photo containing one face."),
    gallery: Path = typer.Option(..., "--gallery", help="Gallery .npz to create or extend."),
    model_dir: Path | None = typer.Option(None, "--model-dir", help="Directory holding the weights."),
) -> None:
    """Add the most prominent face in an image to a gallery."""
    recognizer = _build_recognizer(model_dir=model_dir)
    _load_gallery(recognizer, gallery)
    try:
        recognizer.enroll(name, _read_image(image))
    except AlchemyFaceError as exc:
        typer.echo(f"could not enroll {name}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _save_gallery(recognizer, gallery)
    typer.echo(f"enrolled {name} — gallery now holds {len(recognizer.store)} face(s)")


@app.command()
def identify(
    image: Path = typer.Option(..., "--image", exists=True, dir_okay=False, help="Image to search."),
    gallery: Path = typer.Option(
        ...,
        "--gallery",
        exists=True,
        dir_okay=False,
        help="Gallery .npz to search against.",
    ),
    threshold: float = typer.Option(DEFAULT_THRESHOLD, "--threshold", help="Cosine score required to accept."),
    model_dir: Path | None = typer.Option(None, "--model-dir", help="Directory holding the weights."),
) -> None:
    """Report who each face in an image looks like."""
    recognizer = _build_recognizer(model_dir=model_dir, threshold=threshold)
    recognizer.threshold = threshold
    _load_gallery(recognizer, gallery)
    recognitions = recognizer.identify(_read_image(image))
    if not recognitions:
        typer.echo("no faces detected")
        return
    for recognition in recognitions:
        x, y, w, h = recognition.face.bbox
        if recognition.match is None:
            typer.echo(f"unknown          at ({x},{y},{w},{h})")
        else:
            typer.echo(f"{recognition.match.label:<16} at ({x},{y},{w},{h}) score={recognition.match.score:.3f}")


if __name__ == "__main__":  # pragma: no cover
    app()
