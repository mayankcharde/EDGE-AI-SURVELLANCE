"""Background face detection for the Build tab.

The worker knows nothing about Tk and nothing about OpenCV. It is handed a
callable that turns a path into faces, and hands results back through a queue
that the UI thread drains on a timer. That keeps the concurrency testable with
a fake detector, and keeps the widget free of thread plumbing.

Three problems this exists to solve, all inherited from the original app:

**The image on screen must not wait behind a folder prefetch.** Jobs carry a
priority, and the one the user is looking at is submitted in front.

**Opening a different folder must abandon the old one.** Every job and result
carries a generation; bumping it discards queued work and any result already in
flight, so a slow detection from the previous folder cannot land on the new one.

**A failing image must not take the worker with it.** A detector that raises is
reported as a result with an error, and the loop carries on.
"""

from __future__ import annotations

import itertools
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from alchemyface.types import Face

DetectCallable = Callable[[Path], "list[Face] | None"]
"""Turns a path into faces. ``None`` is accepted and read as "no faces", because
that is what ``cv2``'s detector returns."""

FOREGROUND = 0
BACKGROUND = 1
_STOP = -1
"""Priority of the shutdown sentinel: ahead of every real job."""

DRAIN_LIMIT = 16
"""How many results the UI thread takes per tick, so a large burst cannot
monopolise it."""


@dataclass(frozen=True)
class DetectionResult:
    """What came back for one image."""

    index: int
    generation: int
    faces: list[Face] | None
    """``None`` only when ``error`` is set."""

    error: str | None


@dataclass(order=True)
class _Job:
    priority: int
    sequence: int
    """Breaks priority ties in submission order, and keeps the queue from ever
    comparing the fields below it."""

    index: int = 0
    generation: int = 0
    path: Path | None = None


class DetectionWorker:
    """One background thread detecting faces, with priorities and generations."""

    def __init__(self, *, detect: DetectCallable) -> None:
        self._detect = detect
        self._jobs: queue.PriorityQueue[_Job] = queue.PriorityQueue()
        self._results: queue.Queue[DetectionResult] = queue.Queue()
        self._sequence = itertools.count()
        self._generation = 0
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._lock = threading.Lock()
        self._in_flight: set[tuple[int, int]] = set()
        """(generation, index) pairs currently queued or running, so
        re-submitting the same image while it is pending does not detect it
        twice."""

        self._completed: set[tuple[int, int]] = set()
        """(generation, index) pairs already detected. Needed because a caller
        cannot tell "still queued" from "finished but not yet collected": the
        result sits in a queue the UI thread drains on a timer. Without this,
        a submission landing in that window detects the same image again."""

    # -------------------------------------------------------------- state
    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ----------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Start the thread, or do nothing if it is already running."""
        if self.is_running:
            return
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, name="AlchemyFaceDetect", daemon=True)
        self._thread.start()

    def shutdown(self, wait: bool = False) -> None:
        """Stop the thread. Safe to call more than once."""
        self._stopped = True
        self._jobs.put(_Job(priority=_STOP, sequence=next(self._sequence)))
        if wait and self._thread is not None:
            self._thread.join(timeout=5.0)

    def new_generation(self) -> int:
        """Abandon all queued and in-flight work. Returns the new generation."""
        with self._lock:
            self._generation += 1
            self._in_flight.clear()
            self._completed.clear()
            current = self._generation
        self._clear(self._jobs)
        self._clear(self._results)
        return current

    def forget(self, index: int) -> None:
        """Discard what is known about one image, so it can be detected again.

        Re-detect asks for work on an image already marked complete, which the
        completion record would otherwise skip. Forgetting is deliberately
        explicit: silently allowing repeats would reintroduce the double
        detection the record exists to prevent.
        """
        with self._lock:
            key = (self._generation, index)
            self._completed.discard(key)
            self._in_flight.discard(key)

    @staticmethod
    def _clear(q: queue.Queue[Any]) -> None:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    # -------------------------------------------------------------- submit
    def submit(self, index: int, path: Path, *, foreground: bool = True) -> None:
        """Queue one image for detection.

        A foreground submission is placed ahead of every background one.

        A *background* re-submission of something already queued is dropped, so
        eagerly prefetching a whole folder cannot pile up duplicates. A
        foreground one is deliberately let through, so navigating to an image
        waiting behind a prefetch moves it to the front; the duplicate job is
        skipped in the loop once the first has completed. Anything already
        detected is dropped either way.
        """
        if self._stopped:
            return
        with self._lock:
            key = (self._generation, index)
            if key in self._completed:
                return
            if key in self._in_flight and not foreground:
                return
            # A foreground submission for something already queued is allowed
            # through, so navigating to an image waiting behind a folder
            # prefetch moves it to the front. The duplicate job is skipped in
            # the loop once the first has completed.
            self._in_flight.add(key)
            generation = self._generation
        self._jobs.put(
            _Job(
                priority=FOREGROUND if foreground else BACKGROUND,
                sequence=next(self._sequence),
                index=index,
                generation=generation,
                path=path,
            )
        )

    def drain(self, limit: int = DRAIN_LIMIT) -> list[DetectionResult]:
        """Take up to ``limit`` finished results. Never blocks.

        Results from an abandoned generation are discarded here rather than
        handed to a caller who would have to check for them.
        """
        current = self.generation
        out: list[DetectionResult] = []
        while len(out) < limit:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                break
            if result.generation == current:
                out.append(result)
        return out

    # ---------------------------------------------------------------- loop
    def _loop(self) -> None:
        while True:
            try:
                job = self._jobs.get(timeout=0.25)
            except queue.Empty:
                if self._stopped:
                    return
                continue
            if job.priority == _STOP or self._stopped:
                return
            if job.path is None:
                continue
            with self._lock:
                if job.generation != self._generation:
                    continue  # the folder changed while this waited
                if (job.generation, job.index) in self._completed:
                    continue  # a duplicate raised for priority; already done
            self._run(job)

    def _run(self, job: _Job) -> None:
        assert job.path is not None
        try:
            faces = self._detect(job.path)
        except Exception as exc:  # noqa: BLE001 - any failure is a result, not a crash
            self._finish(job, None, f"{type(exc).__name__}: {exc}")
            return
        self._finish(job, list(faces) if faces else [], None)

    def _finish(self, job: _Job, faces: list[Face] | None, error: str | None) -> None:
        with self._lock:
            key = (job.generation, job.index)
            self._in_flight.discard(key)
            self._completed.add(key)
        self._results.put(
            DetectionResult(
                index=job.index,
                generation=job.generation,
                faces=faces,
                error=error,
            )
        )
