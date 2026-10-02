"""What the Tk on this machine can be trusted with.

Apple ships Tcl/Tk **8.5.9**, released in 2010, as the system Tk, and any Python
built without pointing at something newer links against it. On a current macOS
it does not work: a window opens and paints nothing at all — not even a plain
``tk.Label`` with a background colour — and a full ``update()`` can take the
interpreter down with it.

So the application warns at startup when it finds such a Tk, naming the fix,
and it writes that warning to stderr as well as into the window — the whole
symptom being that nothing in the window can be read.

The threshold lives here so there is one place that knows what "too old" means.
"""

from __future__ import annotations

import tkinter as tk

MIN_DRAWABLE_TK = (8, 6)
"""The oldest Tk that actually renders on a current macOS.

8.5.9, which Apple ships, opens a window and paints nothing at all. It is also
unstable under ``update()``, though that is not what this threshold is for — a
full ``update()`` inside a loop is avoided on *every* Tk version, because it
re-enters the event loop rather than merely redrawing. See
``ResizeView._pump``.
"""


def tk_patchlevel(widget: tk.Misc) -> tuple[int, ...]:
    """The loaded Tk's version, e.g. ``(8, 6, 18)``.

    ``info patchlevel`` rather than ``tkinter.TkVersion``, which is a float and
    so cannot distinguish 8.6.0 from 8.6.18, and rounds 8.5.9 to ``8.5``.
    Returns ``()`` if Tk will not answer, which callers must treat as unknown
    rather than as old — refusing to work because a version could not be read
    would be worse than the risk it is guarding against.
    """
    try:
        raw = str(widget.tk.call("info", "patchlevel"))
    except tk.TclError:
        return ()
    parts: list[int] = []
    for piece in raw.split("."):
        digits = ""
        for char in piece:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def stale_tk_warning(widget: tk.Misc) -> str | None:
    """A message naming the fix, or ``None`` when the Tk in use is fine.

    Written for someone looking at a blank window, so it says what they are
    seeing before it says what to do about it.
    """
    version = tk_patchlevel(widget)
    if not version or version >= MIN_DRAWABLE_TK:
        return None
    shown = ".".join(str(part) for part in version)
    return (
        f"This Python is using Tcl/Tk {shown}, which is too old to draw on a "
        "current macOS — the window will open but is likely to be blank.\n\n"
        "Tk 8.6 or newer is needed. To fix it:\n\n"
        "  brew install tcl-tk@8\n"
        '  PYTHON_CONFIGURE_OPTS="--enable-framework \\\n'
        "    --with-tcltk-includes='-I$(brew --prefix tcl-tk@8)/include/tcl-tk' \\\n"
        "    --with-tcltk-libs='-L$(brew --prefix tcl-tk@8)/lib -ltcl8.6 -ltk8.6'\" \\\n"
        "    pyenv install 3.10.21\n\n"
        "Note that Homebrew's default `tcl-tk` is 9.x, which CPython only "
        "supports from 3.13 — `tcl-tk@8` is the one to install. The library and "
        "the `alchemyface` command-line tools are unaffected."
    )
