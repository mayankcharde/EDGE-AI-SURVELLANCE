"""Reading frames from a camera or a video file.

A thin wrapper over ``cv2.VideoCapture`` that fails loudly when the device
will not open, releases itself on the way out of a ``with`` block, and offers
an iterator so callers do not have to write the read-check-read loop by hand.
"""

from __future__ import annotations

from types import TracebackType
from typing import Iterator

import cv2
import numpy as np
from numpy.typing import NDArray


class VideoSource:
    """A camera index or a path to a video file, as a context manager."""

    def __init__(
        self,
        source: int | str = 0,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"could not open video source {source!r}")
        if width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        self._released = False

    def read(self) -> NDArray[np.uint8] | None:
        """The next frame, or ``None`` once the stream is exhausted."""
        ok, frame = self._capture.read()
        if not ok:
            return None
        return np.asarray(frame, dtype=np.uint8)

    def frames(self) -> Iterator[NDArray[np.uint8]]:
        """Yield frames until the stream ends."""
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def release(self) -> None:
        """Release the device. Safe to call more than once."""
        if not self._released:
            self._capture.release()
            self._released = True

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
