"""Entry point for ``python -m alchemyface``.

The console script installed by pip is the usual way in, but a package that
cannot be run with ``-m`` is a small papercut: it is the first thing a Python
programmer tries, and the application this was ported from was started with a
plain ``python main.py``. This makes ``python -m alchemyface db`` work without
the console script being on PATH — useful from a checkout, or when the script
directory is not on PATH.

Deliberately identical to running ``alchemyface``: same commands, same help.
"""

from alchemyface.cli import app

if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    app()
