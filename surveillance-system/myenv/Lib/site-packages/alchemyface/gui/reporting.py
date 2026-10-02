"""How the application tells the user something happened.

Hard-coding ``tkinter.messagebox`` into the views made them untestable without
constructing real modal dialogs — which, with a worker thread alive, segfaulted.
Reporting is an injected dependency instead: the app uses dialogs, tests use a
recorder, and nothing in a test ever opens a window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class Reporter(Protocol):
    """Somewhere to send a message the user should see."""

    def info(self, title: str, message: str) -> None: ...

    def error(self, title: str, message: str) -> None: ...


class DialogReporter:
    """The real thing: modal message boxes."""

    def info(self, title: str, message: str) -> None:
        from tkinter import messagebox

        messagebox.showinfo(title, message)

    def error(self, title: str, message: str) -> None:
        from tkinter import messagebox

        messagebox.showerror(title, message)


@dataclass
class RecordingReporter:
    """Keeps what it was told, for tests and for headless use."""

    infos: list[tuple[str, str]] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    def info(self, title: str, message: str) -> None:
        self.infos.append((title, message))

    def error(self, title: str, message: str) -> None:
        self.errors.append((title, message))

    @property
    def last_error(self) -> str:
        return self.errors[-1][1] if self.errors else ""

    @property
    def last_info(self) -> str:
        return self.infos[-1][1] if self.infos else ""

    def clear(self) -> None:
        self.infos.clear()
        self.errors.clear()
