"""The Edit DB tab: open an existing database, change it, save it.

A table of entries on the left, an add-faces panel on the right. The widget is
thin: :class:`~alchemyface.gui.edit_data.EditSession` holds the state and
decides what is legal, and this displays it.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable, Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageTk

from alchemyface.errors import AlchemyFaceError
from alchemyface.gui.annotation_data import (
    DEFAULT_GROUP,
    IMAGE_EXTENSIONS,
    default_face_name,
)
from alchemyface.gui.annotation_view import RecognizerLike
from alchemyface.gui.edit_data import EditSession, PendingFace
from alchemyface.gui.reporting import DialogReporter, Reporter

THUMB_SIZE = 48
Anchor = Literal["nw", "n", "ne", "w", "center", "e", "sw", "s", "se"]

TITLE = "Entries"


class ColumnSpec:
    """One column of the entries table."""

    def __init__(self, key: str, heading: str, width: int, anchor: Anchor) -> None:
        self.key = key
        self.heading = heading
        self.width = width
        self.anchor = anchor


class EditDBView(ttk.Frame):
    """Load a database, trim it, extend it, save it."""

    COLUMN_SPECS: tuple[ColumnSpec, ...] = (
        ColumnSpec("idx", "#", 50, "e"),
        ColumnSpec("name", "Name", 180, "w"),
        ColumnSpec("group", "Group", 140, "w"),
        ColumnSpec("dim", "Dim", 70, "center"),
        ColumnSpec("norm", "L2 norm", 90, "e"),
        ColumnSpec("preview", "First values", 320, "w"),
    )
    COLUMNS = tuple(spec.key for spec in COLUMN_SPECS)

    def __init__(
        self,
        parent: tk.Misc,
        *,
        recognizer_provider: Callable[[], RecognizerLike | None],
        group_presets_provider: Callable[[], list[str]],
        on_status: Callable[[str], None],
        on_preset_added: Callable[[str], None] | None = None,
        reporter: Reporter | None = None,
    ) -> None:
        super().__init__(parent)
        self._get_recognizer = recognizer_provider
        self._get_presets = group_presets_provider
        self._set_status = on_status
        self._on_preset_added = on_preset_added or (lambda _g: None)
        self.reporter: Reporter = reporter or DialogReporter()

        self.session = EditSession()
        self._path_var = tk.StringVar(value="")
        self._summary_var = tk.StringVar(value="No DB loaded.")
        self._add_folder_var = tk.StringVar(value="")
        self._add_image_var = tk.StringVar(value="")
        self._pending_vars: list[tk.Variable] = []
        self._thumb_refs: list[ImageTk.PhotoImage] = []
        self._build_ui()

    # ================================================================ public

    @property
    def frame_title(self) -> str:
        """The entries frame's label, carrying ` *` when unsaved."""
        return self._entries_frame.cget("text")

    @property
    def summary_text(self) -> str:
        """The line under the path row, as displayed."""
        return self._summary_var.get()

    def pending_card_count(self) -> int:
        """How many candidate cards are actually drawn.

        Counts widgets rather than model rows, so a pane that was never
        refreshed — cards left over from candidates already consumed — is
        visible to a test instead of hiding behind a correct model.
        """
        return sum(1 for child in self._pending_inner.winfo_children() if child.winfo_class() == "TLabelframe")

    def row_count(self) -> int:
        return len(self._tree.get_children())

    def table_values(self) -> list[tuple[str, ...]]:
        return [tuple(str(v) for v in self._tree.item(item, "values")) for item in self._tree.get_children()]

    def load(self, path: Path | str | None = None) -> bool:
        """Read a database. Returns whether it worked."""
        raw = str(path) if path is not None else self._path_var.get().strip()
        if not raw:
            return self._failed("No file path set.")
        self._path_var.set(raw)
        target = Path(raw)
        if not target.is_file():
            return self._failed(f"File does not exist: {target}")
        try:
            self.session.load(target)
        except AlchemyFaceError as exc:
            return self._failed(str(exc))
        for group in sorted(self.session.groups_in_use()):
            self._on_preset_added(group)
        self._render()
        # load() drops the pending list; without this the cards for the previous
        # database's candidates stay on screen and can still be ticked.
        self._render_pending()
        problems = self.session.problems
        if problems:
            self._set_status(f"Loaded {len(self.session.entries)} entries with {len(problems)} problem(s).")
            self.reporter.info(
                "Loaded with problems",
                f"{target.name} loaded so it can be repaired, but "
                f"{len(problems)} entr{'y is' if len(problems) == 1 else 'ies are'} malformed:\n"
                + "\n".join(f"  • {problem}" for problem in problems)
                + "\n\nDelete those rows before saving — saving validates every vector.",
            )
        else:
            self._set_status(f"Loaded {len(self.session.entries)} entries from {target}")
        return True

    def remove_selected(self, indices: list[int] | None = None) -> int:
        """Drop rows. With no argument, drops whatever the table has selected."""
        wanted = indices if indices is not None else self._selected_indices()
        removed = self.session.remove(wanted)
        if removed:
            self._render()
            self._set_status(f"Removed {removed} entr{'y' if removed == 1 else 'ies'}.")
        return removed

    def set_group(self, index: int, group: str) -> None:
        self.session.set_group(index, group)
        value = group.strip()
        if value and value not in self._get_presets():
            self._on_preset_added(value)
        self._render()

    def process_image(self, path: Path | str) -> int:
        """Detect faces in one image and offer them as candidates."""
        return self._process([Path(path)])

    def process_folder(self, folder: Path | str) -> int:
        """Detect faces in every image in a folder and offer them."""
        images = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            self._failed(f"No images found in {folder}")
            return 0
        return self._process(images)

    def set_pending_include(self, index: int, value: bool) -> None:
        if 0 <= index < len(self.session.pending):
            self.session.pending[index].include = value

    def set_pending_name(self, index: int, value: str) -> None:
        if 0 <= index < len(self.session.pending):
            self.session.pending[index].name = value

    def set_pending_group(self, index: int, value: str) -> None:
        if 0 <= index < len(self.session.pending):
            self.session.pending[index].group = value

    def add_checked(self) -> int:
        """Merge every usable ticked candidate into the table.

        Both panes are redrawn unconditionally. Redrawing only when something
        was added left consumed candidates on screen, so pressing the button a
        second time appeared to do nothing at all.
        """
        result = self.session.merge_checked()
        self._render()
        self._render_pending()
        if result.added and result.skipped:
            self._set_status(f"Added {result.added} face(s); skipped {len(result.skipped)}.")
            self.reporter.info(
                "Added with skips",
                f"Added {result.added} face(s).\n\nSkipped {len(result.skipped)}, still listed below:\n"
                + "\n".join(f"  • {reason}" for reason in result.skipped),
            )
        elif result.added:
            self._set_status(f"Added {result.added} face(s).")
        elif result.skipped:
            self._failed("Nothing was added:\n" + "\n".join(f"  • {reason}" for reason in result.skipped))
        else:
            self._set_status("Nothing ticked — tick a face to add it.")
        return result.added

    def save(self, path: Path | str | None = None) -> bool:
        """Write the database. With no argument, writes where it came from.

        Faces added without loading a database first have nowhere to go, so ask
        rather than refuse — that is a first save, not a mistake.
        """
        if path is None and self.session.path is None and self.session.entries:
            self._save_as()
            return self.session.path is not None
        try:
            destination = self.session.save(path)
        except ValueError as exc:
            return self._failed(str(exc))
        except OSError as exc:
            return self._failed(f"Cannot write the database: {exc}")
        self._render()
        size_kb = destination.stat().st_size / 1024
        self._set_status(f"Saved {len(self.session.entries)} entries → {destination}")
        self.reporter.info(
            "Save successful",
            f"Saved {len(self.session.entries)} entries ({size_kb:.1f} KB) to:\n{destination}",
        )
        return True

    def has_unsaved_changes(self) -> bool:
        return self.session.dirty

    def shutdown(self) -> None:
        """Release Tk variables while the interpreter is still alive."""
        self._release_pending_vars()
        self._thumb_refs = []

    # =============================================================== internals

    def _failed(self, reason: str) -> bool:
        self._set_status(f"Edit DB: {reason}")
        self.reporter.error("Edit DB", reason)
        return False

    def _selected_indices(self) -> list[int]:
        children = list(self._tree.get_children())
        return [children.index(item) for item in self._tree.selection()]

    def _process(self, images: list[Path]) -> int:
        recognizer = self._get_recognizer()
        if recognizer is None:
            self._failed("No model loaded, so faces cannot be detected.")
            return 0
        found: list[PendingFace] = []
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                self._failed(f"Could not read {path.name}.")
                continue
            typed: NDArray[np.uint8] = np.asarray(image, dtype=np.uint8)
            try:
                faces = recognizer.detect(typed)
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                self._failed(f"{path.name}: detection failed — {exc}")
                continue
            for position, face in enumerate(faces):
                try:
                    embedding = recognizer.embed(typed, face)
                except Exception as exc:  # noqa: BLE001
                    self._failed(f"{path.name}: embedding failed — {exc}")
                    continue
                found.append(
                    PendingFace(
                        source=path,
                        face=face,
                        include=True,
                        name=default_face_name(path.stem, position, len(faces)),
                        group=DEFAULT_GROUP,
                        embedding=embedding,
                    )
                )
        if found:
            self.session.add_pending(found)
            self._render_pending()
            self._set_status(f"Found {len(found)} face(s) to add.")
        else:
            # Silence here read as a dead button: the click did nothing visible
            # whether detection ran and found nothing or never ran at all.
            where = images[0].name if len(images) == 1 else f"{len(images)} images"
            self._set_status(f"No faces found in {where}.")
            self.reporter.info("No faces", f"No faces were detected in {where}.")
        return len(found)

    # -------------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0, minsize=340)
        self.rowconfigure(1, weight=1)

        top = ttk.LabelFrame(self, text="Face database", padding=8)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 2))
        ttk.Label(top, text="Path:").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(top, textvariable=self._path_var, width=70).grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(top, text="Browse…", command=self._pick_db).grid(row=0, column=2, padx=6, pady=3)
        ttk.Button(top, text="Load", command=self.load).grid(row=0, column=3, padx=6, pady=3)
        # A line that always says what is loaded and whether it is saved. The
        # status bar is shared with three other tabs, so it scrolls away and
        # cannot be relied on to answer "did my save actually happen".
        ttk.Label(top, textvariable=self._summary_var, foreground="#555555").grid(
            row=1, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 2)
        )
        top.columnconfigure(1, weight=1)

        self._entries_frame = ttk.LabelFrame(self, text=TITLE, padding=4)
        self._entries_frame.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=4)
        self._tree = ttk.Treeview(self._entries_frame, columns=self.COLUMNS, show="headings", selectmode="extended")
        for spec in self.COLUMN_SPECS:
            self._tree.heading(spec.key, text=spec.heading)
            self._tree.column(spec.key, width=spec.width, anchor=spec.anchor)
        bar = ttk.Scrollbar(self._entries_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=bar.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.bind("<Double-1>", self._on_double_click)
        self._tree.bind("<Delete>", lambda _e: self.remove_selected())
        self._tree.bind("<BackSpace>", lambda _e: self.remove_selected())

        actions = ttk.Frame(self, padding=(8, 4))
        actions.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Button(actions, text="Remove selected", command=self.remove_selected).pack(side=tk.LEFT)
        ttk.Button(actions, text="💾  Save", command=lambda: self.save()).pack(side=tk.RIGHT, padx=4)
        ttk.Button(actions, text="Save as…", command=self._save_as).pack(side=tk.RIGHT)

        add = ttk.LabelFrame(self, text="Add faces", padding=6)
        add.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=4)
        # Path box + Browse + Process for each, rather than a Browse that
        # processes immediately: a path could not be typed or pasted, and
        # re-running after fixing a name meant re-opening the dialog.
        ttk.Label(add, text="Folder:").pack(side=tk.TOP, anchor="w")
        folder_row = ttk.Frame(add)
        folder_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
        ttk.Entry(folder_row, textvariable=self._add_folder_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(folder_row, text="…", width=3, command=self._pick_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(add, text="Process folder", command=self._process_folder_clicked).pack(
            side=tk.TOP, fill=tk.X, pady=(0, 6)
        )

        ttk.Label(add, text="Image:").pack(side=tk.TOP, anchor="w")
        image_row = ttk.Frame(add)
        image_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
        ttk.Entry(image_row, textvariable=self._add_image_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(image_row, text="…", width=3, command=self._pick_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(add, text="Process image", command=self._process_image_clicked).pack(
            side=tk.TOP, fill=tk.X, pady=(0, 2)
        )
        ttk.Button(add, text="Add checked", command=self.add_checked).pack(side=tk.TOP, fill=tk.X, pady=(8, 2))
        self._pending_canvas = tk.Canvas(add, highlightthickness=0, width=320)
        pbar = ttk.Scrollbar(add, orient="vertical", command=self._pending_canvas.yview)
        self._pending_canvas.configure(yscrollcommand=pbar.set)
        self._pending_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._pending_inner = ttk.Frame(self._pending_canvas)
        self._pending_window = self._pending_canvas.create_window((0, 0), window=self._pending_inner, anchor="nw")
        self._pending_inner.bind(
            "<Configure>",
            lambda _e: self._pending_canvas.configure(scrollregion=self._pending_canvas.bbox("all")),
        )

    def _pick_db(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Open face database",
            filetypes=[("Pickle", "*.pkl"), ("All files", "*.*")],
        )
        if chosen:
            self._path_var.set(chosen)
            self.load()

    def _pick_folder(self) -> None:
        chosen = filedialog.askdirectory(title="Select a folder of face images")
        if chosen:
            self._add_folder_var.set(chosen)

    def _pick_image(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in sorted(IMAGE_EXTENSIONS))
        chosen = filedialog.askopenfilename(
            title="Select a face image",
            filetypes=[("Images", patterns), ("All files", "*.*")],
        )
        if chosen:
            self._add_image_var.set(chosen)

    def _process_folder_clicked(self) -> None:
        raw = self._add_folder_var.get().strip()
        if not raw:
            self._failed("Set a folder to process.")
            return
        self.process_folder(raw)

    def _process_image_clicked(self) -> None:
        raw = self._add_image_var.get().strip()
        if not raw:
            self._failed("Set an image to process.")
            return
        self.process_image(raw)

    def _save_as(self) -> None:
        chosen = filedialog.asksaveasfilename(
            title="Save face database",
            defaultextension=".pkl",
            filetypes=[("Pickle", "*.pkl"), ("All files", "*.*")],
        )
        if chosen:
            self.save(chosen)

    def _on_double_click(self, event: tk.Event) -> None:
        """Inline-edit the Group cell, as the original did."""
        if self._tree.identify_region(event.x, event.y) != "cell":
            return
        column = self._tree.identify_column(event.x)
        if column != f"#{self.COLUMNS.index('group') + 1}":
            return
        item = self._tree.identify_row(event.y)
        if not item:
            return
        index = list(self._tree.get_children()).index(item)
        box = self._tree.bbox(item, column)
        if not box:
            return
        current = self.session.entries[index].group
        var = tk.StringVar(value=current)
        combo = ttk.Combobox(self._tree, textvariable=var, values=list(self._get_presets()))
        combo.place(x=box[0], y=box[1], width=box[2], height=box[3])
        combo.focus_set()

        cancelled = False

        def commit(_event: tk.Event | None = None) -> None:
            if cancelled:
                return
            combo.destroy()
            self.set_group(index, var.get())

        def cancel(_event: tk.Event | None = None) -> None:
            """Escape abandons the edit, leaving the original group in place."""
            nonlocal cancelled
            cancelled = True
            combo.destroy()

        def commit_unless_dropdown(_event: tk.Event | None = None) -> None:
            # Opening the dropdown takes focus away from the entry, which fired
            # FocusOut and committed the half-typed value while the list was
            # still open. Committing only when focus left the widget entirely
            # is what the original did.
            if combo.winfo_exists() and str(combo.tk.call("focus")).startswith(str(combo)):
                return
            commit()

        combo.bind("<Return>", commit)
        combo.bind("<Escape>", cancel)
        combo.bind("<FocusOut>", commit_unless_dropdown)
        combo.bind("<<ComboboxSelected>>", commit)

    # --------------------------------------------------------------- render

    def _refresh_summary(self) -> None:
        """Keep the line under the path row telling the truth.

        Called from every render, so an edit that marks the session dirty shows
        up without each caller having to remember.
        """
        count = len(self.session.entries)
        if self.session.path is None and not count:
            self._summary_var.set("No DB loaded.")
        elif self.session.dirty:
            self._summary_var.set(f"{count} entries (unsaved changes)")
        elif self.session.path is not None:
            size = self.session.path.stat().st_size / 1024 if self.session.path.exists() else 0.0
            self._summary_var.set(f"{count} entries · {size:.1f} KB · {self.session.path}")
        else:
            self._summary_var.set(f"{count} entries")

    def _render(self) -> None:
        self._refresh_summary()
        for item in self._tree.get_children():
            self._tree.delete(item)
        for row in self.session.rows():
            self._tree.insert(
                "",
                tk.END,
                values=(
                    row.index,
                    row.name,
                    row.group,
                    row.dim,
                    f"{row.norm:.4f}",
                    row.preview,
                ),
            )
        self._entries_frame.configure(text=f"{TITLE}{self.session.title_suffix}")
        self._refresh_summary()

    def _release_pending_vars(self) -> None:
        """Detach traces before dropping, so Tk collects them while it can."""
        for variable in self._pending_vars:
            try:
                for modes, name in variable.trace_info():
                    for mode in modes:
                        variable.trace_remove(mode, name)
            except tk.TclError:
                pass
        self._pending_vars = []

    def _render_pending(self) -> None:
        for child in self._pending_inner.winfo_children():
            child.destroy()
        self._release_pending_vars()
        self._thumb_refs = []
        if not self.session.pending:
            ttk.Label(self._pending_inner, text="(nothing to add)", padding=8).pack()
            return
        for index, face in enumerate(self.session.pending):
            self._build_card(index, face)

    def _build_card(self, index: int, face: PendingFace) -> None:
        frame = ttk.LabelFrame(self._pending_inner, text=f"  {face.source.name}  ", padding=6)
        frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=3)

        header = ttk.Frame(frame)
        header.pack(side=tk.TOP, fill=tk.X)
        thumb = self._thumbnail(face)
        if thumb is not None:
            tk.Label(header, image=thumb, borderwidth=1, relief="solid").pack(side=tk.LEFT, padx=4)
            self._thumb_refs.append(thumb)
        include_var = tk.BooleanVar(value=face.include)
        ttk.Checkbutton(header, text="Include", variable=include_var).pack(side=tk.LEFT, padx=4)

        grid = ttk.Frame(frame)
        grid.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        ttk.Label(grid, text="Name").grid(row=0, column=0, sticky="w", padx=(0, 4))
        name_var = tk.StringVar(value=face.name)
        ttk.Entry(grid, textvariable=name_var).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(grid, text="Group").grid(row=1, column=0, sticky="w", padx=(0, 4))
        group_var = tk.StringVar(value=face.group)
        ttk.Combobox(grid, textvariable=group_var, values=list(self._get_presets())).grid(
            row=1, column=1, sticky="ew", pady=2
        )
        grid.columnconfigure(1, weight=1)

        self._pending_vars.extend((include_var, name_var, group_var))
        include_var.trace_add("write", lambda *_a: self.set_pending_include(index, include_var.get()))
        name_var.trace_add("write", lambda *_a: self.set_pending_name(index, name_var.get()))
        group_var.trace_add("write", lambda *_a: self.set_pending_group(index, group_var.get()))

    def _thumbnail(self, face: PendingFace) -> ImageTk.PhotoImage | None:
        image = cv2.imread(str(face.source))
        if image is None:
            return None
        x, y, w, h = face.face.bbox
        x, y = max(0, x), max(0, y)
        x2 = min(image.shape[1], x + w)
        y2 = min(image.shape[0], y + h)
        if x2 <= x or y2 <= y:
            return None
        crop = image[y:y2, x:x2][:, :, ::-1]
        pil = Image.fromarray(crop).resize((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil)
