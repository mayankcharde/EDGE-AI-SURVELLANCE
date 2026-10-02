"""The Build tab: three panes for turning a folder of photos into a database.

Sidebar of images on the left, the current image with numbered face boxes in the
middle, one panel per face on the right.

The widget is thin on purpose. Geometry, status and validation live in
:mod:`alchemyface.gui.annotation_data`; the threading lives in
:mod:`alchemyface.gui.detect_worker`. What is here is Tk: building widgets,
drawing the canvas, and turning callbacks into changes on the model.
"""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageTk

from alchemyface.gui.annotation_data import (
    BOX_COLOURS,
    DEFAULT_GROUP,
    IMAGE_EXTENSIONS,
    BgrCache,
    EntryStatus,
    FaceAnnotation,
    ImageEntry,
    default_face_name,
    face_at,
    fit_image,
    sidebar_colour,
    sidebar_text,
)
from alchemyface.gui.detect_worker import DetectionWorker
from alchemyface.types import Face

THUMB_SIZE = 56
POLL_MS = 50
"""How often the Tk thread collects finished detections."""

RESIZE_DEBOUNCE_MS = 80
"""Window drags fire a storm of Configure events; coalesce them."""

PREFETCH_AHEAD = 3


class RecognizerLike(Protocol):
    """What the view needs of a recognizer. Anything conforming will do, which
    is how the tests substitute a fake and never load a model."""

    def detect(self, image: NDArray[np.uint8]) -> list[Face]: ...

    def embed(self, image: NDArray[np.uint8], face: Face) -> NDArray[np.float32]: ...


class AnnotationView(ttk.Frame):
    """Three-pane annotation of a folder of face photos."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        recognizer_provider: Callable[[], RecognizerLike | None],
        ensure_recognizer: Callable[[], RecognizerLike | None] | None = None,
        group_presets_provider: Callable[[], list[str]],
        on_status: Callable[[str], None],
        on_preset_added: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._get_recognizer = recognizer_provider
        self._ensure_recognizer = ensure_recognizer or recognizer_provider
        """Two providers, because they run on different threads.
        ``recognizer_provider`` is called from the worker and must never touch
        Tk; ``ensure_recognizer`` is called from here and may load and report."""
        self._get_presets = group_presets_provider
        self._set_status = on_status
        self._on_preset_added = on_preset_added or (lambda _g: None)

        self._entries: list[ImageEntry] = []
        self._current = -1
        self._selected_face: int | None = None
        self._cache = BgrCache()
        self._worker = DetectionWorker(detect=self._detect_path)

        # canvas render state
        self._transform = fit_image(1, 1, 1, 1)
        self._photo: ImageTk.PhotoImage | None = None
        self._photo_key: tuple[Path, int, int] | None = None
        self._box_items: list[int] = []
        self._thumb_refs: list[ImageTk.PhotoImage] = []
        self._face_vars: list[tk.Variable] = []
        """Every Tk variable bound to a face card, so its trace can be detached
        when the card is replaced.

        A trace registers the variable inside Tk, so destroying the widget does
        not free it. Re-rendering the face panel on every navigation therefore
        accumulates three Tcl variables per face indefinitely — measured at 480
        after 80 re-renders of a two-face image."""
        self._resize_job: str | None = None
        self._poll_job: str | None = None

        self._counter_var = tk.StringVar(value="—")
        self._label_var = tk.StringVar(value="(no image loaded)")

        self._build_ui()

    # ============================================================ public

    @property
    def entries(self) -> list[ImageEntry]:
        return self._entries

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def current_index(self) -> int:
        return self._current

    @property
    def selected_face(self) -> int | None:
        return self._selected_face

    @property
    def worker_is_running(self) -> bool:
        return self._worker.is_running

    def sidebar_labels(self) -> list[str]:
        return [self._listbox.get(i) for i in range(self._listbox.size())]

    def load_folder(self, folder: Path) -> bool:
        """Read a folder, list it, and detect every image in the background.

        Refuses without a model rather than starting: every image would fail,
        and the sidebar would fill with what looks like "no face detected".
        """
        if self._ensure_recognizer() is None:
            self._set_status("No model loaded, so nothing can be detected.")
            return False
        # Checked before anything is cleared. Opening the wrong folder by
        # mistake used to discard a whole session's naming work and replace it
        # with an empty list, which is not what "no images here" should cost.
        images = sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not images and self._entries:
            self._set_status(f"No images in {folder.name} — keeping the {len(self._entries)} already open.")
            return False
        self._worker.new_generation()
        self._cache.clear()
        self._photo_key = None
        self._photo = None
        self._entries = [ImageEntry(path=path) for path in images]
        self._current = -1
        self._selected_face = None

        self._listbox.delete(0, tk.END)
        for entry in self._entries:
            self._listbox.insert(tk.END, sidebar_text(entry))
            self._listbox.itemconfigure(tk.END, foreground=sidebar_colour(entry))
        self._counter_var.set(f"0 / {len(self._entries)}")
        self._render_canvas()
        self._render_faces()

        if not self._entries:
            self._label_var.set("(no images)")
            self._set_status("No images found in that folder.")
            return True

        self._worker.start()
        self._start_polling()
        self.select_index(0)
        # Eagerly queue the whole folder so the sidebar fills in even if the
        # user never navigates. The current image was submitted in front.
        for index, entry in enumerate(self._entries):
            self._worker.submit(index, entry.path, foreground=False)
        self._set_status(f"Loaded {len(self._entries)} images — detecting in the background.")
        return True

    def wait_until_settled(self, timeout: float = 10.0) -> bool:
        """Collect finished detections until every image has one.

        Only useful without a running mainloop, which is the situation in tests:
        the `after` timer that normally drains the worker never fires.

        It drains directly and then flushes drawing with ``update_idletasks``,
        rather than calling ``update``. A full ``update`` also runs the pending
        `after` callback, so every result would be collected and rendered twice
        — and on macOS's system Tk 8.5.9 that churn of ``ImageTk.PhotoImage``
        segfaults occasionally. CI runs these tests on Tk 8.6, where the
        problem does not arise.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._collect()
            self.update_idletasks()
            if not self._entries or all(e.detected for e in self._entries):
                return True
            time.sleep(0.005)
        return False

    def select_index(self, index: int) -> None:
        """Show one image, detecting it first if need be, and prefetch ahead."""
        if not 0 <= index < len(self._entries):
            return
        self._current = index
        self._selected_face = None
        entry = self._entries[index]
        self._counter_var.set(f"{index + 1} / {len(self._entries)}")
        self._label_var.set(entry.path.name)
        self._listbox.selection_clear(0, tk.END)
        self._listbox.selection_set(index)
        self._listbox.activate(index)
        self._listbox.see(index)

        if not entry.detected:
            self._worker.submit(index, entry.path, foreground=True)
        for ahead in range(1, PREFETCH_AHEAD + 1):
            nxt = index + ahead
            if nxt < len(self._entries) and not self._entries[nxt].detected:
                self._worker.submit(nxt, self._entries[nxt].path, foreground=False)

        self._render_canvas()
        self._render_faces()

    def next_image(self) -> None:
        if self._current < len(self._entries) - 1:
            self.select_index(self._current + 1)

    def previous_image(self) -> None:
        if self._current > 0:
            self.select_index(self._current - 1)

    def set_include(self, face_index: int, value: bool) -> None:
        face = self._face(face_index)
        if face is None or face.include == value:
            return
        face.include = value
        self._mark_edited()
        self._refresh_row(self._current)
        self._render_canvas()
        self._render_faces()

    def set_name(self, face_index: int, value: str) -> None:
        face = self._face(face_index)
        if face is None or face.name == value:
            return
        face.name = value
        self._mark_edited()

    def set_group(self, face_index: int, value: str) -> None:
        face = self._face(face_index)
        if face is None or face.group == value:
            return
        face.group = value
        self._mark_edited()
        stripped = value.strip()
        if stripped and stripped not in self._get_presets():
            self._on_preset_added(stripped)

    def select_face_at_image_point(self, x: float, y: float) -> int | None:
        """Select whichever face box contains an image-space point."""
        if self._current < 0:
            return None
        found = face_at(self._entries[self._current].faces, x, y)
        self._selected_face = found
        self._render_canvas()
        return found

    def redetect(self, confirm: Callable[[], bool] | None = None) -> None:
        """Detect the current image again.

        ``confirm`` is only consulted when the entry has unsaved edits, so the
        common case does not nag. Returning False cancels.
        """
        if self._current < 0:
            return
        entry = self._entries[self._current]
        if entry.edited and confirm is not None and not confirm():
            return
        # Checked before anything is cleared: this wipes the faces and their
        # typed names, and the worker cannot put them back without a model.
        if self._ensure_recognizer() is None:
            self._set_status(f"{entry.path.name}: no model loaded, so nothing was re-detected.")
            return
        entry.detected = False
        entry.faces = []
        entry.error = None
        entry.edited = False
        self._selected_face = None
        self._cache.drop(entry.path)
        self._photo_key = None
        # The worker remembers what it has already detected, so it has to be
        # told to forget this image before it will look at it again.
        self._worker.forget(self._current)
        self._worker.submit(self._current, entry.path, foreground=True)
        self._refresh_row(self._current)
        self._render_canvas()
        self._render_faces()
        self._set_status(f"{entry.path.name}: re-detecting…")

    def invalidate_embeddings(self) -> None:
        """Forget every cached embedding, because the model changed."""
        count = 0
        for entry in self._entries:
            for face in entry.faces:
                if face.embedding is not None:
                    face.embedding = None
                    count += 1
        if count:
            self._set_status(f"Model changed — {count} cached embedding(s) will be recomputed.")

    def fill_embeddings(self) -> list[str]:
        """Compute any missing embedding for an included face.

        Returns the problems encountered, so the caller can refuse to save.
        """
        # The *loading* provider, deliberately. This runs on the main thread —
        # its only caller is the Save button — and changing a model path drops
        # the recognizer, so the non-loading provider made Save refuse for good
        # with "no model loaded", one line after promising a recompute.
        recognizer = self._ensure_recognizer()
        errors: list[str] = []
        for entry in self._entries:
            if not entry.detected:
                continue
            for index, face in enumerate(entry.faces):
                if not face.include or face.embedding is not None:
                    continue
                if recognizer is None:
                    errors.append(f"{entry.path.name}: no model loaded.")
                    continue
                image = self._read_bgr(entry.path)
                if image is None:
                    errors.append(f"{entry.path.name}: could not re-read the image.")
                    continue
                try:
                    face.embedding = recognizer.embed(image, face.face)
                except Exception as exc:  # noqa: BLE001 - reported, not raised
                    errors.append(f"{entry.path.name}: face #{index + 1} embed failed — {exc}")
        return errors

    def shutdown(self) -> None:
        """Stop the worker, cancel timers, release Tk variables.

        Safe to call repeatedly. The variables are dropped here so they are
        collected while the interpreter still exists, rather than after it has
        been torn down.
        """
        # Wait, rather than signalling and walking away. A worker outliving its
        # view holds references to Tk objects that are about to be destroyed,
        # and touching those from another thread segfaults.
        self._worker.shutdown(wait=True)
        self._release_face_vars()
        self._thumb_refs = []
        self._photo = None
        for attr in ("_poll_job", "_resize_job"):
            job = getattr(self, attr)
            if job is not None:
                try:
                    self.after_cancel(job)
                except (tk.TclError, ValueError):
                    pass
                setattr(self, attr, None)

    # ============================================================= internals

    def _face(self, face_index: int) -> FaceAnnotation | None:
        if self._current < 0:
            return None
        faces = self._entries[self._current].faces
        return faces[face_index] if 0 <= face_index < len(faces) else None

    def _mark_edited(self) -> None:
        if 0 <= self._current < len(self._entries):
            self._entries[self._current].edited = True

    def _read_bgr(self, path: Path) -> NDArray[np.uint8] | None:
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        image = cv2.imread(str(path))
        if image is None:
            return None
        typed = np.asarray(image, dtype=np.uint8)
        self._cache.put(path, typed)
        return typed

    def _detect_path(self, path: Path) -> list[Face] | None:
        """Runs on the worker thread."""
        recognizer = self._get_recognizer()
        if recognizer is None:
            raise RuntimeError("no model loaded")
        image = self._read_bgr(path)
        if image is None:
            raise RuntimeError("could not read the image")
        return recognizer.detect(image)

    # ------------------------------------------------------------- polling

    def _start_polling(self) -> None:
        if self._poll_job is None:
            self._poll_job = self.after(POLL_MS, self._poll)

    def _poll(self) -> None:
        self._poll_job = None
        self._collect()
        if self._worker.is_running:
            self._poll_job = self.after(POLL_MS, self._poll)

    def _collect(self) -> None:
        for result in self._worker.drain():
            if not 0 <= result.index < len(self._entries):
                continue
            entry = self._entries[result.index]
            entry.detected = True
            entry.edited = False
            entry.faces = []
            entry.error = result.error
            if result.error is not None:
                self._set_status(f"{entry.path.name}: {result.error}")
            else:
                total = len(result.faces or [])
                for position, face in enumerate(result.faces or []):
                    entry.faces.append(
                        FaceAnnotation(
                            face=face,
                            include=True,
                            name=default_face_name(entry.path.stem, position, total),
                            group=DEFAULT_GROUP,
                        )
                    )
            self._refresh_row(result.index)
            if result.index == self._current:
                self._render_canvas()
                self._render_faces()
                if result.error is None:
                    # The status stayed on "detecting in the background" even
                    # after the image on screen had finished, so there was no
                    # way to tell a still-running detection from one that found
                    # nothing.
                    count = len(entry.faces)
                    self._set_status(f"{entry.path.name}: {count} face(s) detected.")

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0, minsize=240)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=0, minsize=320)
        self.rowconfigure(0, weight=1)

        sidebar = ttk.LabelFrame(self, text="Images", padding=4)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=4)
        self._listbox = tk.Listbox(sidebar, exportselection=False, activestyle="dotbox")
        bar = ttk.Scrollbar(sidebar, orient="vertical", command=self._listbox.yview)
        self._listbox.configure(yscrollcommand=bar.set)
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        centre = ttk.Frame(self)
        centre.grid(row=0, column=1, sticky="nsew", padx=2, pady=4)
        top = ttk.Frame(centre)
        top.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        ttk.Button(top, text="🔄 Re-detect", command=self._on_redetect_clicked).pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self._label_var).pack(side=tk.LEFT, padx=10)
        self._canvas = tk.Canvas(centre, background="#1e1e1e", highlightthickness=0)
        self._canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<Button-1>", self._on_canvas_click)

        nav = ttk.Frame(centre)
        nav.pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))
        ttk.Button(nav, text="◀ Prev", command=self.previous_image).pack(side=tk.LEFT)
        ttk.Label(nav, textvariable=self._counter_var, width=14, anchor="center").pack(side=tk.LEFT, padx=8)
        ttk.Button(nav, text="Next ▶", command=self.next_image).pack(side=tk.LEFT)

        right = ttk.LabelFrame(self, text="Faces", padding=2)
        right.grid(row=0, column=2, sticky="nsew", padx=(2, 4), pady=4)
        self._right_canvas = tk.Canvas(right, highlightthickness=0, width=320)
        rbar = ttk.Scrollbar(right, orient="vertical", command=self._right_canvas.yview)
        self._right_canvas.configure(yscrollcommand=rbar.set)
        self._right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._right_inner = ttk.Frame(self._right_canvas)
        self._right_window = self._right_canvas.create_window((0, 0), window=self._right_inner, anchor="nw")
        self._right_inner.bind(
            "<Configure>",
            lambda _e: self._right_canvas.configure(scrollregion=self._right_canvas.bbox("all")),
        )
        self._right_canvas.bind(
            "<Configure>",
            lambda e: self._right_canvas.itemconfigure(self._right_window, width=e.width),
        )

    def _on_listbox_select(self, _event: tk.Event) -> None:
        selection = self._listbox.curselection()
        if selection and selection[0] != self._current:
            self.select_index(selection[0])

    def _on_redetect_clicked(self) -> None:
        from tkinter import messagebox

        self.redetect(
            confirm=lambda: messagebox.askyesno(
                "Re-detect",
                "This image has unsaved edits. Re-detecting will discard them. Continue?",
            )
        )

    def _refresh_row(self, index: int) -> None:
        if not 0 <= index < len(self._entries):
            return
        entry = self._entries[index]
        self._listbox.delete(index)
        self._listbox.insert(index, sidebar_text(entry))
        self._listbox.itemconfigure(index, foreground=sidebar_colour(entry))
        if index == self._current:
            self._listbox.selection_clear(0, tk.END)
            self._listbox.selection_set(index)

    # --------------------------------------------------------------- canvas

    def _on_canvas_configure(self, _event: tk.Event) -> None:
        if self._resize_job is not None:
            try:
                self.after_cancel(self._resize_job)
            except (tk.TclError, ValueError):
                pass
        self._resize_job = self.after(RESIZE_DEBOUNCE_MS, self._render_canvas)

    def _on_canvas_click(self, event: tk.Event) -> None:
        image_x, image_y = self._transform.to_image(event.x, event.y)
        self.select_face_at_image_point(image_x, image_y)

    def _render_canvas(self) -> None:
        self._resize_job = None
        self._canvas.delete("all")
        self._box_items = []
        if self._current < 0:
            return
        entry = self._entries[self._current]
        width = max(1, self._canvas.winfo_width())
        height = max(1, self._canvas.winfo_height())

        image = self._read_bgr(entry.path)
        if image is None:
            self._canvas.create_text(width // 2, height // 2, text="(could not read image)", fill="#bbbbbb")
            return

        self._transform = fit_image(image.shape[1], image.shape[0], width, height)
        key = (entry.path, width, height)
        if key != self._photo_key or self._photo is None:
            rgb = image[:, :, ::-1]
            pil = Image.fromarray(rgb).resize((self._transform.width, self._transform.height), Image.Resampling.LANCZOS)
            self._photo = ImageTk.PhotoImage(pil)
            self._photo_key = key
        self._canvas.create_image(
            self._transform.offset_x,
            self._transform.offset_y,
            anchor="nw",
            image=self._photo,
        )

        if entry.status is EntryStatus.PENDING:
            self._canvas.create_text(
                self._transform.offset_x + self._transform.width // 2,
                self._transform.offset_y + 24,
                text="Detecting…",
                fill="#ffd700",
                font=("TkDefaultFont", 16, "bold"),
            )
            return

        for position, face in enumerate(entry.faces):
            colour = BOX_COLOURS[position % len(BOX_COLOURS)]
            x, y, w, h = face.bbox
            x1, y1 = self._transform.to_canvas(x, y)
            x2, y2 = self._transform.to_canvas(x + w, y + h)
            self._box_items.append(
                self._canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    outline=colour,
                    width=4 if position == self._selected_face else 2,
                    dash=() if face.include else (4, 3),
                )
            )
            # Drawn on a filled chip in the box's own colour, not as bare
            # coloured text: over a light photo the number was unreadable, and
            # the colour is what ties it to the row in the face panel.
            label_x = x1 + 4
            label_y = max(self._transform.offset_y, y1 - 18)
            self._box_items.append(
                self._canvas.create_rectangle(
                    label_x - 2,
                    label_y - 2,
                    label_x + 24,
                    label_y + 14,
                    fill=colour,
                    outline=colour,
                )
            )
            self._box_items.append(
                self._canvas.create_text(
                    label_x,
                    label_y,
                    text=f"#{position + 1}",
                    anchor="nw",
                    fill="black",
                    font=("TkDefaultFont", 9, "bold"),
                )
            )

    # ---------------------------------------------------------- face panel

    def _release_face_vars(self) -> None:
        """Detach traces, then drop the variables.

        A trace registers the variable inside Tk, so merely dropping the list
        leaves it referenced and it is collected at some arbitrary later moment
        — possibly after the interpreter has gone, when ``Variable.__del__``
        raises "main thread is not in main loop". Removing the trace first lets
        it be collected here, while Tk is definitely alive.
        """
        for variable in self._face_vars:
            try:
                # trace_info yields (modes, callback) where modes is a tuple,
                # so each mode is removed individually.
                for modes, name in variable.trace_info():
                    for mode in modes:
                        variable.trace_remove(mode, name)
            except tk.TclError:
                pass  # the interpreter is already gone; nothing to detach
        self._face_vars = []

    def _render_faces(self) -> None:
        for child in self._right_inner.winfo_children():
            child.destroy()
        self._thumb_refs = []
        self._release_face_vars()
        if self._current < 0:
            ttk.Label(self._right_inner, text="(no image)", padding=8).pack()
            return
        entry = self._entries[self._current]
        if entry.status is EntryStatus.PENDING:
            ttk.Label(self._right_inner, text="(detecting…)", padding=8).pack()
            return
        if entry.status is EntryStatus.FAILED:
            ttk.Label(
                self._right_inner,
                text=f"Detection failed:\n{entry.error}",
                padding=8,
                foreground="#cc2200",
                wraplength=290,
            ).pack()
            return
        if not entry.faces:
            ttk.Label(
                self._right_inner,
                text="No face detected in this image.",
                padding=8,
                foreground="#cc7700",
            ).pack()
            return

        for position, face in enumerate(entry.faces):
            self._build_face_card(position, face, entry)

    def _build_face_card(self, position: int, face: FaceAnnotation, entry: ImageEntry) -> None:
        colour = BOX_COLOURS[position % len(BOX_COLOURS)]
        frame = ttk.LabelFrame(self._right_inner, text=f"  Face #{position + 1}  ", padding=6)
        frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=3)

        header = ttk.Frame(frame)
        header.pack(side=tk.TOP, fill=tk.X)
        tk.Label(header, text="■", fg=colour, font=("TkDefaultFont", 14)).pack(side=tk.LEFT)
        thumb = self._thumbnail(entry, face)
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

        # Register them, or _release_face_vars has nothing to release. This was
        # missing until an audit measured 480 leaked Tcl variables over 80
        # re-renders — the mechanism existed, the one line wiring it did not.
        self._face_vars.extend((include_var, name_var, group_var))
        include_var.trace_add("write", lambda *_a: self.set_include(position, include_var.get()))
        name_var.trace_add("write", lambda *_a: self.set_name(position, name_var.get()))
        group_var.trace_add("write", lambda *_a: self.set_group(position, group_var.get()))
        frame.bind("<Button-1>", lambda _e: self._select_face(position))

    def _select_face(self, position: int) -> None:
        self._selected_face = position
        self._render_canvas()

    def _thumbnail(self, entry: ImageEntry, face: FaceAnnotation) -> ImageTk.PhotoImage | None:
        image = self._read_bgr(entry.path)
        if image is None:
            return None
        x, y, w, h = face.bbox
        x, y = max(0, x), max(0, y)
        x2 = min(image.shape[1], x + w)
        y2 = min(image.shape[0], y + h)
        if x2 <= x or y2 <= y:
            return None
        crop = image[y:y2, x:x2][:, :, ::-1]
        pil = Image.fromarray(crop).resize((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil)
