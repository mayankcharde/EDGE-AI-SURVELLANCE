"""The Resize tab: shrink photos until the detector can see the face.

The work itself is in :mod:`alchemyface.gui.resize_data`; this is the Tk around
it. Source and output each accept either a folder or a single image, so the two
rows are symmetric.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable

from alchemyface.gui.annotation_data import IMAGE_EXTENSIONS
from alchemyface.gui.reporting import DialogReporter, Reporter
from alchemyface.gui.resize_data import (
    DEFAULT_RATIO,
    MAX_RATIO,
    MIN_RATIO,
    default_output_folder,
    plan_folder,
    resize_one,
    resize_one_outcome,
)


class ResizeView(ttk.Frame):
    """Bulk or single-image resizing, with a per-file log."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_status: Callable[[str], None],
        reporter: Reporter | None = None,
    ) -> None:
        super().__init__(parent)
        self._set_status = on_status
        self.reporter: Reporter = reporter or DialogReporter()

        self._folder_in = tk.StringVar(value="")
        self._folder_out = tk.StringVar(value="")
        self._image_in = tk.StringVar(value="")
        self._image_out = tk.StringVar(value="")
        self._ratio_var = tk.DoubleVar(value=DEFAULT_RATIO)
        self._folder_out_chosen = False
        self._image_out_chosen = False
        self._build_ui()

    # ================================================================ public

    @property
    def ratio(self) -> float:
        """The ratio currently in the box, with a non-numeric entry read as the
        default. Not clamped: a value outside the legal range is refused when a
        run starts, so silently reporting a different number here would
        contradict what the button does."""
        return self._safe_ratio()

    def log_text(self) -> str:
        return self._log.get("1.0", tk.END)

    def clear_log(self) -> None:
        self._log.configure(state=tk.NORMAL)
        self._log.delete("1.0", tk.END)

    def resize_folder_now(self, source: Path | str | None = None, destination: Path | str | None = None) -> int:
        """Resize a whole folder. Returns how many files were written."""
        raw_src = str(source) if source is not None else self._folder_in.get().strip()
        # Checked as text, before Path() sees it: Path("") is PosixPath("."),
        # which passes is_dir() and then failed deep inside with an empty name.
        if not raw_src:
            return self._failed("Set a source folder.") and 0
        src = Path(raw_src)
        if not src.is_dir():
            return self._failed(f"Not a folder: {src}") and 0
        ratio = self._checked_ratio()
        if ratio is None:
            return 0
        raw_out = str(destination) if destination is not None else self._folder_out.get().strip()
        dst = Path(raw_out) if raw_out else default_output_folder(src)
        self._folder_in.set(str(src))
        self._folder_out.set(str(dst))

        try:
            plan = plan_folder(src, dst, ratio)
        except (ValueError, NotADirectoryError) as exc:
            return self._failed(str(exc)) and 0
        if not plan:
            self._failed(f"No images found in {src}")
            return 0

        self._begin_run(f"Resizing {len(plan)} image(s) at ratio {ratio:.2f}", src, dst)
        written = 0
        failed = 0
        # Resized one at a time rather than in a single call so each line is
        # visible as it happens. A folder of a few hundred photos took long
        # enough that a log filled in at the end looked like a hung window.
        for source_path, destination_path in plan:
            outcome = resize_one_outcome(source_path, destination_path, ratio)
            self._append(str(outcome))
            written += 1 if outcome.ok else 0
            failed += 0 if outcome.ok else 1
            self._pump()
        self._append("")
        self._append(f"Done. {written} resized, {failed} failed.")
        self._set_status(f"Resized {written} of {len(plan)} image(s) at {ratio:.2f} → {dst}")
        return written

    def resize_image_now(self, source: Path | str | None = None, destination: Path | str | None = None) -> bool:
        """Resize a single image. Returns whether it worked."""
        raw_src = str(source) if source is not None else self._image_in.get().strip()
        if not raw_src:
            return self._failed("Set a source image.")
        src = Path(raw_src)
        if not src.is_file():
            return self._failed(f"Not a file: {src}")
        if src.suffix.lower() not in IMAGE_EXTENSIONS:
            # Refused up front rather than handed to Pillow, which would happily
            # write a .tif the Build tab then ignores when it scans the folder.
            expected = ", ".join(sorted(IMAGE_EXTENSIONS))
            return self._failed(f"Unsupported file extension: {src.suffix}. Expected one of {expected}.")
        ratio = self._checked_ratio()
        if ratio is None:
            return False
        raw_out = str(destination) if destination is not None else self._image_out.get().strip()
        dst = Path(raw_out) if raw_out else src.with_name(f"{src.stem}_resized{src.suffix}")
        self._image_in.set(str(src))
        self._image_out.set(str(dst))

        self._begin_run(f"Resizing 1 image at ratio {ratio:.2f}", src, dst)
        try:
            result = resize_one(src, dst, ratio)
        except ValueError as exc:
            return self._failed(str(exc))
        except OSError as exc:
            return self._failed(f"{src.name}: {exc}")
        self._append(f"{src.name}: {result}")
        self._append("")
        self._append("Done. 1 resized, 0 failed.")
        self._set_status(f"Resized {src.name} at {ratio:.2f} → {dst}")
        return True

    def shutdown(self) -> None:
        """Nothing to release; present so the window can call it uniformly."""

    # ============================================================== internals

    def _safe_ratio(self) -> float:
        try:
            return float(self._ratio_var.get())
        except (tk.TclError, ValueError):
            return DEFAULT_RATIO

    def _checked_ratio(self) -> float | None:
        """The ratio to use, or ``None`` having already reported why not.

        Silently clamping an out-of-range ratio meant a typo — 50 for 0.5 —
        rewrote a folder of photos at a size the user never asked for, with
        nothing on screen to say so. A resize cannot be undone, so it asks.
        """
        try:
            raw = float(self._ratio_var.get())
        except (tk.TclError, ValueError):
            self._failed("Ratio must be a number.")
            return None
        if raw != raw or not MIN_RATIO <= raw <= MAX_RATIO:
            self._failed(f"Ratio must be between {MIN_RATIO} and {MAX_RATIO}. Got {raw:g}.")
            return None
        return raw

    def _begin_run(self, header: str, source: Path, destination: Path) -> None:
        """Clear the log and write the run header, as the original did.

        Without the clear, a second run appended under the first and it was not
        obvious which lines belonged to which.
        """
        self.clear_log()
        self._append(header)
        self._append(f"  src: {source}")
        self._append(f"  dst: {destination}")
        self._append("")

    def _pump(self) -> None:
        """Redraw the log so each line appears as the file is written.

        ``update_idletasks()``, where the original called the full ``update()``.
        That difference was first made because ``update()`` segfaulted this
        project's GUI suite on Apple's system Tcl/Tk 8.5.9, and the guess was
        that a newer Tk would make it safe again.

        It does not. Restoring ``update()`` on Tk 8.6.18 deadlocks the resize
        suite instead of crashing it, because the hazard was never the Tk
        version: ``update()`` processes *every* pending event, not just redraws,
        so calling it inside this loop re-enters whatever else is scheduled —
        the detection worker's polling among it. Old Tk turned that
        re-entrancy into a segfault and new Tk turns it into a hang.

        So idle tasks it is, on every Tk. The redraw still happens, so progress
        is visible as each file lands; what is given up is handling clicks
        mid-run, so a long batch cannot be cancelled until it finishes.
        """
        try:
            self.update_idletasks()
        except tk.TclError:
            pass  # the window went away mid-run

    def _failed(self, reason: str) -> bool:
        self._append(f"ERROR: {reason}")
        self._set_status(f"Resize: {reason}")
        self.reporter.error("Resize", reason)
        return False

    def _append(self, line: str) -> None:
        self._log.configure(state=tk.NORMAL)
        self._log.insert(tk.END, line + "\n")
        self._log.see(tk.END)

    # -------------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        source = ttk.LabelFrame(self, text="Source", padding=8)
        source.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(6, 2))
        self._path_row(source, 0, "Folder:", self._folder_in, "Browse folder…", self._pick_folder_in)
        self._path_row(source, 1, "Image:", self._image_in, "Browse image…", self._pick_image_in)
        source.columnconfigure(1, weight=1)

        output = ttk.LabelFrame(self, text="Output", padding=8)
        output.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        self._path_row(output, 0, "Folder:", self._folder_out, "Browse folder…", self._pick_folder_out)
        self._path_row(output, 1, "Image:", self._image_out, "Save as…", self._pick_image_out)
        output.columnconfigure(1, weight=1)

        controls = ttk.Frame(self, padding=(8, 4))
        controls.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(controls, text="Ratio:").pack(side=tk.LEFT)
        ttk.Spinbox(
            controls,
            from_=MIN_RATIO,
            to=MAX_RATIO,
            increment=0.05,
            textvariable=self._ratio_var,
            width=6,
            format="%.2f",
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(
            controls,
            text="  (0.5 = half · 0.25 = quarter · 2.0 = double)",
            foreground="#666666",
        ).pack(side=tk.LEFT)

        actions = ttk.Frame(self, padding=(8, 4))
        actions.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(actions, text="Resize folder", command=lambda: self.resize_folder_now()).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Resize image", command=lambda: self.resize_image_now()).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Clear log", command=self.clear_log).pack(side=tk.RIGHT, padx=4)

        frame = ttk.LabelFrame(self, text="Log", padding=4)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)
        self._log = tk.Text(frame, height=12, wrap="none", font=("TkFixedFont", 10))
        vertical = ttk.Scrollbar(frame, orient="vertical", command=self._log.yview)
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=self._log.xview)
        self._log.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self._log.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

    def _path_row(
        self,
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        variable: tk.StringVar,
        button: str,
        command: Callable[[], None],
    ) -> None:
        ttk.Label(parent, text=label, width=8).grid(row=row, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(parent, text=button, command=command).grid(row=row, column=2, padx=6, pady=3)

    # ---------------------------------------------------------- file pickers

    def _pick_folder_in(self) -> None:
        chosen = filedialog.askdirectory(title="Select source folder")
        if not chosen:
            return
        self._folder_in.set(chosen)
        if not self._folder_out_chosen:
            self._folder_out.set(str(default_output_folder(chosen)))

    def _pick_folder_out(self) -> None:
        chosen = filedialog.askdirectory(title="Select output folder")
        if chosen:
            self._folder_out.set(chosen)
            self._folder_out_chosen = True

    def _pick_image_in(self) -> None:
        # Both cases: a camera that writes .JPG is common enough that filtering
        # on the lowercase set alone hid the file the user came to pick.
        patterns = " ".join(f"*{ext}" for ext in sorted(IMAGE_EXTENSIONS))
        patterns += " " + " ".join(f"*{ext.upper()}" for ext in sorted(IMAGE_EXTENSIONS))
        chosen = filedialog.askopenfilename(
            title="Select an image", filetypes=[("Images", patterns), ("All files", "*.*")]
        )
        if not chosen:
            return
        self._image_in.set(chosen)
        if not self._image_out_chosen:
            path = Path(chosen)
            self._image_out.set(str(path.with_name(f"{path.stem}_resized{path.suffix}")))

    def _pick_image_out(self) -> None:
        current = self._image_out.get().strip() or self._image_in.get().strip()
        suggested = Path(current) if current else Path.cwd() / "resized.jpg"
        patterns = " ".join(f"*{ext}" for ext in sorted(IMAGE_EXTENSIONS))
        chosen = filedialog.asksaveasfilename(
            title="Save resized image as",
            defaultextension=suggested.suffix or ".jpg",
            filetypes=[("Images", patterns), ("All files", "*.*")],
            initialdir=str(suggested.parent),
            initialfile=suggested.name,
        )
        if chosen:
            self._image_out.set(chosen)
            self._image_out_chosen = True
