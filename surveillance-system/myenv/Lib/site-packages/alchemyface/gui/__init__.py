"""The Face DB Builder desktop application.

``App`` is resolved lazily. Importing this package must not pull in ``tkinter``,
which on Debian and Ubuntu is a separate OS package (``python3-tk``): the pure
presentation helpers next door — :mod:`alchemyface.gui.inspect_data` and
friends — have to be importable, and testable, wherever Python runs.

    from alchemyface.gui import App     # imports tkinter, needs it installed
    from alchemyface.gui import inspect_data   # numpy only
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from alchemyface.gui.app import App

__all__ = ["App"]


def __getattr__(name: str) -> Any:
    if name == "App":
        from alchemyface.gui.app import App as _App

        return _App
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
