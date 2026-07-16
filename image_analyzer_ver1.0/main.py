"""画像解析ツール — 統計・ヒストグラム・ROI・プロファイル・比較・品質・FFT・バッチ。"""

from __future__ import annotations

import csv
import json
import random
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.analysis import (  # noqa: E402
    COLORMAPS,
    assess_quality,
    circle_mask,
    compare_images,
    compute_fft_magnitude,
    compute_histogram,
    compute_stats,
    extract_line_profile,
    polygon_mask,
    rect_mask,
    stats_row_for_export,
    to_float_gray,
)
from module.file_tree import (  # noqa: E402
    DirNode,
    build_dir_node,
    directory_has_images,
    format_dir_number,
    format_file_number,
)
from module.image_loader import (  # noqa: E402
    DEFAULT_RAW_SETTINGS,
    RAW_DTYPE_OPTIONS,
    guess_raw_settings,
    is_supported_image,
    load_image_file,
)
from module.visualization import (  # noqa: E402
    abs_diff_heatmap,
    array_to_display_image,
    overlay_roi_on_image,
    render_fft_image,
    render_histogram_image,
    render_profile_image,
)


RESIZE_METHODS: dict[str, Image.Resampling] = {
    "NEAREST（最近傍）": Image.Resampling.NEAREST,
    "BILINEAR（双線形）": Image.Resampling.BILINEAR,
    "BICUBIC（双三次）": Image.Resampling.BICUBIC,
    "LANCZOS（高品質）": Image.Resampling.LANCZOS,
    "BOX（ボックス平均）": Image.Resampling.BOX,
    "HAMMING（Hamming窓）": Image.Resampling.HAMMING,
}
DEFAULT_RESIZE_METHOD = "LANCZOS（高品質）"
BLOCK_MEAN_RESIZE_METHOD = "BOX（ボックス平均）"


def default_downloads_dir() -> Path:
    downloads = Path.home() / "Downloads"
    if downloads.is_dir():
        return downloads
    return Path.home()


class RawSettingsDialog(tk.Toplevel):
    """RAWファイル読み込み時のパラメータ入力ダイアログ。"""

    def __init__(self, parent: tk.Misc, path: Path, defaults: dict[str, object]) -> None:
        super().__init__(parent)
        self.title("RAW読み込み設定")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result: dict[str, object] | None = None
        file_size = path.stat().st_size

        frame = ttk.Frame(self, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text=f"ファイル: {path.name}").grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 4))
        ttk.Label(frame, text=f"ファイルサイズ: {file_size:,} bytes").grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 8)
        )

        self.width_var = tk.StringVar(value=str(defaults.get("width", 1600)))
        self.height_var = tk.StringVar(value=str(defaults.get("height", 1300)))
        self.channels_var = tk.StringVar(value=str(defaults.get("channels", 1)))
        self.offset_var = tk.StringVar(value=str(defaults.get("offset", 0)))
        self.dtype_var = tk.StringVar(value=str(defaults.get("dtype", "uint16")))
        self.endian_var = tk.StringVar(value=str(defaults.get("endian", "little")))

        fields = [
            ("幅 (width)", self.width_var),
            ("高さ (height)", self.height_var),
            ("チャンネル数", self.channels_var),
            ("オフセット (bytes)", self.offset_var),
        ]
        for row, (label, var) in enumerate(fields, start=2):
            ttk.Label(frame, text=label, width=18).grid(row=row, column=0, sticky=tk.W, pady=3)
            ttk.Entry(frame, textvariable=var, width=16).grid(row=row, column=1, sticky=tk.W, pady=3)

        ttk.Label(frame, text="データ型").grid(row=6, column=0, sticky=tk.W, pady=3)
        ttk.Combobox(
            frame,
            textvariable=self.dtype_var,
            values=[name for name, _ in RAW_DTYPE_OPTIONS],
            state="readonly",
            width=14,
        ).grid(row=6, column=1, sticky=tk.W, pady=3)

        ttk.Label(frame, text="バイトオーダー").grid(row=7, column=0, sticky=tk.W, pady=3)
        ttk.Combobox(
            frame,
            textvariable=self.endian_var,
            values=["little", "big"],
            state="readonly",
            width=14,
        ).grid(row=7, column=1, sticky=tk.W, pady=3)

        self.hint_label = ttk.Label(frame, text="", foreground="#555555")
        self.hint_label.grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))
        self._update_hint(file_size)

        for var in (self.width_var, self.height_var, self.channels_var, self.offset_var, self.dtype_var):
            var.trace_add("write", lambda *_args, size=file_size: self._update_hint(size))

        buttons = ttk.Frame(frame)
        buttons.grid(row=9, column=0, columnspan=2, sticky=tk.E, pady=(12, 0))
        ttk.Button(buttons, text="キャンセル", command=self._cancel).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="読み込み", command=self._confirm).pack(side=tk.RIGHT, padx=(0, 8))

        self.bind("<Escape>", lambda _event: self._cancel())
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _parse_positive_int(self, name: str, var: tk.StringVar) -> int:
        text = var.get().strip()
        if not text.isdigit():
            raise ValueError(f"{name}には0以上の整数を入力してください。")
        return int(text)

    def _update_hint(self, file_size: int) -> None:
        try:
            width = self._parse_positive_int("幅", self.width_var)
            height = self._parse_positive_int("高さ", self.height_var)
            channels = max(1, self._parse_positive_int("チャンネル数", self.channels_var))
            offset = self._parse_positive_int("オフセット", self.offset_var)
            dtype_name = self.dtype_var.get()
            dtype = np.dtype(next(dt for name, dt in RAW_DTYPE_OPTIONS if name == dtype_name))
            expected = width * height * channels * dtype.itemsize + offset
            self.hint_label.config(text=f"想定サイズ: {expected:,} bytes")
        except Exception:
            self.hint_label.config(text="入力値を確認してください。")

    def _confirm(self) -> None:
        try:
            self.result = {
                "width": self._parse_positive_int("幅", self.width_var),
                "height": self._parse_positive_int("高さ", self.height_var),
                "channels": max(1, self._parse_positive_int("チャンネル数", self.channels_var)),
                "offset": self._parse_positive_int("オフセット", self.offset_var),
                "dtype": self.dtype_var.get(),
                "endian": self.endian_var.get(),
            }
        except ValueError as exc:
            messagebox.showerror("入力エラー", str(exc), parent=self)
            return
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


class ZoomableCanvas(ttk.Frame):
    """ズーム・パン・描画操作対応の画像キャンバス。"""

    PAN_SPEED = 0.5

    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_cursor_move: Callable[..., None] | None = None,
        on_drag: Callable[..., None] | None = None,
        on_drag_end: Callable[..., None] | None = None,
        on_click: Callable[..., None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.on_cursor_move = on_cursor_move
        self.on_drag = on_drag
        self.on_drag_end = on_drag_end
        self.on_click = on_click

        self.canvas = tk.Canvas(self, background="#1e1e1e", highlightthickness=0)
        h_scroll = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        h_scroll.grid(row=1, column=0, sticky="ew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._image: Image.Image | None = None
        self._source_array: np.ndarray | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._zoom = 1.0
        self._pan_start: tuple[int, int] | None = None
        self._pan_remainder_x = 0.0
        self._pan_remainder_y = 0.0
        self._drag_start: tuple[int, int] | None = None
        self.interaction_mode = "none"  # none / rect / circle / freehand / line

        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)
        self.canvas.bind("<Control-ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<Control-B1-Motion>", self._on_pan_move)
        self.canvas.bind("<Control-ButtonRelease-1>", self._on_pan_end)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())

    def set_interaction_mode(self, mode: str) -> None:
        self.interaction_mode = mode

    def set_image(self, image: Image.Image | None, source_array: np.ndarray | None = None) -> None:
        self._image = image
        self._source_array = source_array
        self._redraw()

    def get_image(self) -> Image.Image | None:
        return self._image

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.05, min(32.0, zoom))
        self._redraw()

    def get_zoom(self) -> float:
        return self._zoom

    def fit_to_window(self) -> None:
        if self._image is None:
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        zx = cw / self._image.width
        zy = ch / self._image.height
        self._zoom = min(zx, zy, 1.0)
        self._redraw()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def actual_size(self) -> None:
        self._zoom = 1.0
        self._redraw()

    def canvas_to_image(self, canvas_x: int, canvas_y: int) -> tuple[int, int] | None:
        if self._image is None:
            return None
        cx = self.canvas.canvasx(canvas_x)
        cy = self.canvas.canvasy(canvas_y)
        ix = int(cx / self._zoom)
        iy = int(cy / self._zoom)
        ix = max(0, min(ix, self._image.width - 1))
        iy = max(0, min(iy, self._image.height - 1))
        return ix, iy

    def get_pixel_value(self, x: int, y: int) -> str:
        if self._source_array is not None:
            arr = self._source_array
            if arr.ndim == 2:
                return f"{arr[y, x]}"
            if arr.ndim >= 3:
                vals = arr[y, x]
                if np.ndim(vals) == 0:
                    return f"{vals}"
                return ", ".join(str(v) for v in np.asarray(vals).ravel()[:4])
        if self._image is None:
            return "-"
        px = self._image.getpixel((x, y))
        if isinstance(px, int):
            return str(px)
        return ", ".join(str(v) for v in px)

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self._image is None:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2 or 200,
                self.canvas.winfo_height() // 2 or 150,
                text="画像を開いてください",
                fill="#aaaaaa",
                font=("", 12),
            )
            return

        disp_w = max(1, int(round(self._image.width * self._zoom)))
        disp_h = max(1, int(round(self._image.height * self._zoom)))
        if abs(self._zoom - 1.0) < 1e-6:
            disp = self._image
        else:
            disp = self._image.resize((disp_w, disp_h), Image.Resampling.NEAREST)
        self._photo = ImageTk.PhotoImage(disp)
        self.canvas.create_image(0, 0, image=self._photo, anchor=tk.NW)
        self.canvas.config(scrollregion=(0, 0, disp_w, disp_h))

    def _on_mousewheel(self, event: tk.Event) -> None:
        if self._image is None:
            return
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        old_zoom = self._zoom
        self._zoom = max(0.05, min(32.0, self._zoom * factor))
        if abs(self._zoom - old_zoom) < 1e-6:
            return
        self._redraw()

    def _on_pan_start(self, event: tk.Event) -> None:
        self._pan_start = (event.x, event.y)
        self._pan_remainder_x = 0.0
        self._pan_remainder_y = 0.0
        self.canvas.config(cursor="fleur")

    def _on_pan_move(self, event: tk.Event) -> None:
        if self._pan_start is None:
            return
        dx = (event.x - self._pan_start[0]) * self.PAN_SPEED
        dy = (event.y - self._pan_start[1]) * self.PAN_SPEED
        self._pan_start = (event.x, event.y)
        self._pan_remainder_x -= dx
        self._pan_remainder_y -= dy
        scroll_x = int(self._pan_remainder_x)
        scroll_y = int(self._pan_remainder_y)
        self._pan_remainder_x -= scroll_x
        self._pan_remainder_y -= scroll_y
        if scroll_x:
            self.canvas.xview_scroll(scroll_x, "units")
        if scroll_y:
            self.canvas.yview_scroll(scroll_y, "units")

    def _on_pan_end(self, _event: tk.Event) -> None:
        self._pan_start = None
        self.canvas.config(cursor="arrow")

    def _on_motion(self, event: tk.Event) -> None:
        if self.on_cursor_move is None or self._image is None:
            return
        pt = self.canvas_to_image(event.x, event.y)
        if pt is None:
            return
        self.on_cursor_move(pt[0], pt[1], event)

    def _on_leave(self, _event: tk.Event) -> None:
        if self.on_cursor_move is not None:
            self.on_cursor_move(-1, -1, None)

    def _on_left_press(self, event: tk.Event) -> None:
        if self._image is None:
            return
        pt = self.canvas_to_image(event.x, event.y)
        if pt is None:
            return
        if self.interaction_mode == "none":
            if self.on_click is not None:
                self.on_click(pt[0], pt[1])
            return
        self._drag_start = pt
        if self.on_drag is not None:
            self.on_drag(pt[0], pt[1], pt[0], pt[1], "start")

    def _on_left_drag(self, event: tk.Event) -> None:
        if self._drag_start is None or self._image is None or self.on_drag is None:
            return
        pt = self.canvas_to_image(event.x, event.y)
        if pt is None:
            return
        self.on_drag(self._drag_start[0], self._drag_start[1], pt[0], pt[1], "move")

    def _on_left_release(self, event: tk.Event) -> None:
        if self._drag_start is None or self._image is None:
            return
        pt = self.canvas_to_image(event.x, event.y)
        start = self._drag_start
        self._drag_start = None
        if pt is None:
            return
        if self.on_drag_end is not None and self.interaction_mode != "none":
            self.on_drag_end(start[0], start[1], pt[0], pt[1])
        elif self.on_click is not None and self.interaction_mode == "none":
            self.on_click(pt[0], pt[1])


class ImageAnalyzerApp:
    """画像解析ツールのメインアプリケーション。"""

    BROWSE_MODES = ("順番", "ランダム", "番号指定")
    VIEW_MODES = ("通常", "カラーマップ", "FFT", "差分ヒートマップ")

    def __init__(self, root: tk.Tk, initial_paths: list[Path] | None = None) -> None:
        self.root = root
        self.root.title("画像解析ツール")
        self.root.minsize(1400, 900)

        self.file_list: list[Path] = []
        self.root_sources: list[Path] = []
        self._file_tree_iids: dict[int, str] = {}
        self.current_index = -1
        self.source_array: np.ndarray | None = None
        self.working_array: np.ndarray | None = None  # リサイズ後（描画・対話解析用）
        self.base_image: Image.Image | None = None
        self.current_path: Path | None = None
        self.current_file_type: str | None = None
        self.raw_settings: dict[str, object] = dict(DEFAULT_RAW_SETTINGS)
        self.export_dir = default_downloads_dir()

        self.browse_mode = tk.StringVar(value="順番")
        self.index_var = tk.StringVar(value="0001")
        self.view_mode = tk.StringVar(value="通常")
        self.cmap_var = tk.StringVar(value="viridis")
        # 1〜10 → x0.1〜x1.0（step 0.1）
        self.resize_var = tk.IntVar(value=10)
        self.resize_method_var = tk.StringVar(value=DEFAULT_RESIZE_METHOD)
        self.hist_channel = tk.StringVar(value="gray")
        self.hist_bins = tk.IntVar(value=256)
        self.show_cdf = tk.BooleanVar(value=True)
        self.roi_mode = tk.StringVar(value="矩形")
        self.tool_mode = tk.StringVar(value="なし")  # なし / ROI / ライン
        self._resize_job: str | None = None
        self.resize_value_label: ttk.Label | None = None

        self.roi_rect: tuple[int, int, int, int] | None = None
        self.roi_circle: tuple[int, int, float] | None = None
        self.roi_polygon: list[tuple[int, int]] = []
        self.line_points: tuple[int, int, int, int] | None = None
        self.roi_mask: np.ndarray | None = None

        self.compare_array: np.ndarray | None = None
        self.compare_path: Path | None = None
        self.compare_result: dict | None = None
        self._drag_preview: tuple | None = None

        self._plot_photo: ImageTk.PhotoImage | None = None
        self._profile_photo: ImageTk.PhotoImage | None = None
        self._fft_photo: ImageTk.PhotoImage | None = None
        self._compare_photo: ImageTk.PhotoImage | None = None
        self._batch_busy = False

        self._build_ui()
        self._bind_shortcuts()
        self._sync_interaction_mode()

        if initial_paths:
            self._load_paths(initial_paths)

    # ---------- UI ----------
    def _build_ui(self) -> None:
        self._build_menu()
        self._build_toolbar()

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        left = ttk.Frame(main, padding=4, width=280)
        main.add(left, weight=0)
        center = ttk.Frame(main, padding=4)
        main.add(center, weight=3)
        right = ttk.Frame(main, padding=4, width=420)
        main.add(right, weight=1)

        self._build_file_panel(left)
        self._build_view_panel(center)
        self._build_analysis_panel(right)

        self.status_var = tk.StringVar(value="準備完了")
        ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W, padding=(8, 4)).pack(
            fill=tk.X, side=tk.BOTTOM
        )

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ファイル", menu=file_menu)
        file_menu.add_command(label="ファイルを開く...", command=self.open_files, accelerator="Ctrl+O")
        file_menu.add_command(label="フォルダを開く...", command=self.open_directory, accelerator="Ctrl+Shift+O")
        file_menu.add_command(label="ファイルを追加...", command=self.add_files)
        file_menu.add_separator()
        file_menu.add_command(label="比較画像を開く...", command=self.open_compare_image)
        file_menu.add_separator()
        file_menu.add_command(label="解析画像を保存...", command=self.save_display_image)
        file_menu.add_command(label="保存先フォルダを変更...", command=self.choose_export_dir)
        file_menu.add_separator()
        file_menu.add_command(label="終了", command=self.root.quit)

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="表示", menu=view_menu)
        view_menu.add_command(label="ウィンドウに合わせる", command=self.fit_view, accelerator="F")
        view_menu.add_command(label="実サイズ (100%)", command=self.actual_size_view, accelerator="0")
        view_menu.add_command(label="拡大", command=lambda: self.adjust_zoom(1.25), accelerator="+")
        view_menu.add_command(label="縮小", command=lambda: self.adjust_zoom(0.8), accelerator="-")

        nav_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="移動", menu=nav_menu)
        nav_menu.add_command(label="前へ", command=self.show_prev, accelerator="Left")
        nav_menu.add_command(label="次へ", command=self.show_next, accelerator="Right")
        nav_menu.add_command(label="ランダム", command=self.show_random, accelerator="R")
        nav_menu.add_command(label="先頭", command=self.show_first, accelerator="Home")
        nav_menu.add_command(label="末尾", command=self.show_last, accelerator="End")

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ヘルプ", menu=help_menu)
        help_menu.add_command(label="ショートカット一覧", command=self.show_shortcuts_help)

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(fill=tk.X)

        ttk.Button(bar, text="ファイルを開く", command=self.open_files).pack(side=tk.LEFT)
        ttk.Button(bar, text="フォルダを開く", command=self.open_directory).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(bar, text="追加", command=self.add_files).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(bar, text="RAW再読込", command=self.reload_raw).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(bar, text="◀", width=3, command=self.show_prev).pack(side=tk.LEFT)
        ttk.Button(bar, text="▶", width=3, command=self.show_next).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Button(bar, text="🎲", width=3, command=self.show_random).pack(side=tk.LEFT, padx=(2, 0))

        ttk.Label(bar, text="移動:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Combobox(bar, textvariable=self.browse_mode, values=self.BROWSE_MODES, state="readonly", width=10).pack(
            side=tk.LEFT
        )
        ttk.Label(bar, text="番号").pack(side=tk.LEFT, padx=(6, 2))
        index_entry = ttk.Entry(bar, textvariable=self.index_var, width=6)
        index_entry.pack(side=tk.LEFT)
        index_entry.bind("<Return>", lambda _e: self.jump_to_index())
        ttk.Button(bar, text="移動", command=self.jump_to_index).pack(side=tk.LEFT, padx=(2, 0))

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(bar, text="Fit", command=self.fit_view).pack(side=tk.LEFT)
        ttk.Button(bar, text="100%", command=self.actual_size_view).pack(side=tk.LEFT, padx=(2, 0))

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(bar, text="リサイズ").pack(side=tk.LEFT)
        self.resize_value_label = ttk.Label(bar, text="1.0x", width=5)
        self.resize_value_label.pack(side=tk.LEFT, padx=(4, 2))
        ttk.Scale(
            bar,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            variable=self.resize_var,
            command=self._on_resize_slider,
            length=120,
        ).pack(side=tk.LEFT)
        ttk.Combobox(
            bar,
            textvariable=self.resize_method_var,
            values=list(RESIZE_METHODS.keys()),
            state="readonly",
            width=18,
        ).pack(side=tk.LEFT, padx=(4, 0))
        self.resize_method_var.trace_add("write", lambda *_: self._on_resize_method_change())

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(bar, text="表示:").pack(side=tk.LEFT)
        ttk.Combobox(
            bar,
            textvariable=self.view_mode,
            values=self.VIEW_MODES,
            state="readonly",
            width=14,
        ).pack(side=tk.LEFT, padx=(4, 0))
        self.view_mode.trace_add("write", lambda *_: self.refresh_view())

        ttk.Label(bar, text="CMAP").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Combobox(
            bar,
            textvariable=self.cmap_var,
            values=list(COLORMAPS),
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT)
        self.cmap_var.trace_add("write", lambda *_: self.refresh_view())

        self.path_label = ttk.Label(bar, text="ファイル未選択")
        self.path_label.pack(side=tk.RIGHT, padx=(8, 0))

    def _build_file_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="ファイル一覧", font=("", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(
            parent,
            text="D0001=フォルダ(青)  0001=ファイル(緑)",
            foreground="#555555",
            font=("", 8),
        ).pack(anchor=tk.W, pady=(2, 0))

        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 4))

        self.file_tree = ttk.Treeview(list_frame, show="tree", selectmode="browse")
        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=scroll.set)
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_tree.tag_configure("dir", foreground="#1565c0")
        self.file_tree.tag_configure("file", foreground="#2e7d32")
        self.file_tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.file_tree.bind("<Double-1>", self._on_tree_activate)

        jump_row = ttk.Frame(parent)
        jump_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(jump_row, text="ファイル番号").pack(side=tk.LEFT)
        panel_index_entry = ttk.Entry(jump_row, textvariable=self.index_var, width=8)
        panel_index_entry.pack(side=tk.LEFT, padx=(6, 4))
        panel_index_entry.bind("<Return>", lambda _e: self.jump_to_index())
        ttk.Button(jump_row, text="移動", command=self.jump_to_index).pack(side=tk.LEFT)

        ttk.Label(parent, text="統計情報", font=("", 10, "bold")).pack(anchor=tk.W, pady=(4, 0))
        self.meta_text = tk.Text(parent, height=16, width=34, state=tk.DISABLED, font=("Consolas", 9))
        self.meta_text.pack(fill=tk.BOTH, expand=False, pady=(2, 0))

    def _build_view_panel(self, parent: ttk.Frame) -> None:
        tool_row = ttk.Frame(parent)
        tool_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(tool_row, text="描画ツール:").pack(side=tk.LEFT)
        for label, value in (("なし", "なし"), ("ROI", "ROI"), ("ライン", "ライン")):
            ttk.Radiobutton(
                tool_row,
                text=label,
                value=value,
                variable=self.tool_mode,
                command=self._sync_interaction_mode,
            ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(tool_row, text="ROI形状:").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Combobox(
            tool_row,
            textvariable=self.roi_mode,
            values=("矩形", "円", "自由選択"),
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT)
        self.roi_mode.trace_add("write", lambda *_: self._sync_interaction_mode())
        ttk.Button(tool_row, text="ROIクリア", command=self.clear_roi).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(tool_row, text="ラインクリア", command=self.clear_line).pack(side=tk.LEFT, padx=(4, 0))

        view_frame = ttk.LabelFrame(parent, text="プレビュー", padding=2)
        view_frame.pack(fill=tk.BOTH, expand=True)
        self.main_canvas = ZoomableCanvas(
            view_frame,
            on_cursor_move=self._on_main_cursor_move,
            on_drag=self._on_canvas_drag,
            on_drag_end=self._on_canvas_drag_end,
            on_click=self._on_canvas_click,
        )
        self.main_canvas.pack(fill=tk.BOTH, expand=True)

        hint = ttk.Label(
            parent,
            text="ホイール=ズーム / 中・右ドラッグ or Ctrl+左=パン / ROI・ラインは描画ツールで指定",
            foreground="#555555",
            font=("", 8),
        )
        hint.pack(anchor=tk.W, pady=(4, 0))

    def _build_analysis_panel(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        tab_hist = ttk.Frame(notebook, padding=6)
        tab_roi = ttk.Frame(notebook, padding=6)
        tab_profile = ttk.Frame(notebook, padding=6)
        tab_compare = ttk.Frame(notebook, padding=6)
        tab_quality = ttk.Frame(notebook, padding=6)
        tab_fft = ttk.Frame(notebook, padding=6)
        tab_batch = ttk.Frame(notebook, padding=6)

        notebook.add(tab_hist, text="ヒストグラム")
        notebook.add(tab_roi, text="ROI")
        notebook.add(tab_profile, text="プロファイル")
        notebook.add(tab_compare, text="比較")
        notebook.add(tab_quality, text="品質")
        notebook.add(tab_fft, text="FFT")
        notebook.add(tab_batch, text="バッチ")

        self._build_hist_tab(tab_hist)
        self._build_roi_tab(tab_roi)
        self._build_profile_tab(tab_profile)
        self._build_compare_tab(tab_compare)
        self._build_quality_tab(tab_quality)
        self._build_fft_tab(tab_fft)
        self._build_batch_tab(tab_batch)

    def _build_hist_tab(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)
        ttk.Label(row, text="チャンネル").pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=self.hist_channel,
            values=("gray", "rgb"),
            state="readonly",
            width=8,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="bins").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Spinbox(row, from_=16, to=1024, textvariable=self.hist_bins, width=6).pack(side=tk.LEFT)
        ttk.Checkbutton(row, text="CDF", variable=self.show_cdf).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(row, text="更新", command=self.update_histogram).pack(side=tk.RIGHT)

        self.hist_label = ttk.Label(parent)
        self.hist_label.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        ttk.Button(parent, text="ヒストグラム画像を保存", command=self.save_histogram_image).pack(fill=tk.X, pady=(8, 0))

    def _build_roi_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text="描画ツールで ROI を指定すると、ROI 内の統計が表示されます。",
            wraplength=380,
        ).pack(anchor=tk.W)
        self.roi_text = tk.Text(parent, height=18, state=tk.DISABLED, font=("Consolas", 9))
        self.roi_text.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        ttk.Button(parent, text="ROI統計を再計算", command=self.update_roi_stats).pack(fill=tk.X, pady=(8, 0))

    def _build_profile_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text="描画ツール「ライン」で始点→終点をドラッグしてください。",
            wraplength=380,
        ).pack(anchor=tk.W)
        self.profile_label = ttk.Label(parent)
        self.profile_label.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.profile_info = ttk.Label(parent, text="ライン未設定", foreground="#555555")
        self.profile_info.pack(anchor=tk.W, pady=(4, 0))
        ttk.Button(parent, text="プロファイル画像を保存", command=self.save_profile_image).pack(fill=tk.X, pady=(8, 0))

    def _build_compare_tab(self, parent: ttk.Frame) -> None:
        ttk.Button(parent, text="比較画像を開く...", command=self.open_compare_image).pack(fill=tk.X)
        self.compare_path_label = ttk.Label(parent, text="比較画像: 未設定", wraplength=380, foreground="#555555")
        self.compare_path_label.pack(anchor=tk.W, pady=(6, 0))
        ttk.Button(parent, text="比較を実行", command=self.run_compare).pack(fill=tk.X, pady=(8, 0))
        self.compare_text = tk.Text(parent, height=10, state=tk.DISABLED, font=("Consolas", 9))
        self.compare_text.pack(fill=tk.BOTH, expand=False, pady=(8, 0))
        self.compare_label = ttk.Label(parent)
        self.compare_label.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        ttk.Label(parent, text="差分ヒートマップは上部「表示」で切替できます。", foreground="#555555").pack(
            anchor=tk.W, pady=(4, 0)
        )

    def _build_quality_tab(self, parent: ttk.Frame) -> None:
        ttk.Button(parent, text="品質チェック実行", command=self.update_quality).pack(fill=tk.X)
        self.quality_text = tk.Text(parent, height=22, state=tk.DISABLED, font=("Consolas", 9))
        self.quality_text.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

    def _build_fft_tab(self, parent: ttk.Frame) -> None:
        ttk.Button(parent, text="FFTを計算", command=self.update_fft).pack(fill=tk.X)
        ttk.Label(parent, text="プレビュー表示モード「FFT」でも確認できます。", foreground="#555555").pack(
            anchor=tk.W, pady=(4, 0)
        )
        self.fft_label = ttk.Label(parent)
        self.fft_label.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        ttk.Button(parent, text="FFT画像を保存", command=self.save_fft_image).pack(fill=tk.X, pady=(8, 0))

    def _build_batch_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text="現在のファイル一覧すべてに対して統計・品質を算出し、CSV / JSON に保存します。",
            wraplength=380,
        ).pack(anchor=tk.W)
        self.batch_include_quality = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="品質指標も含める", variable=self.batch_include_quality).pack(
            anchor=tk.W, pady=(8, 0)
        )
        ttk.Button(parent, text="CSVで一括出力...", command=lambda: self.run_batch_export("csv")).pack(
            fill=tk.X, pady=(12, 0)
        )
        ttk.Button(parent, text="JSONで一括出力...", command=lambda: self.run_batch_export("json")).pack(
            fill=tk.X, pady=(6, 0)
        )
        self.batch_status = ttk.Label(parent, text="", foreground="#555555", wraplength=380)
        self.batch_status.pack(anchor=tk.W, pady=(12, 0))

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-o>", lambda _e: self.open_files())
        self.root.bind("<Control-O>", lambda _e: self.open_directory())
        self.root.bind("<Left>", lambda _e: self.show_prev())
        self.root.bind("<Right>", lambda _e: self.show_next())
        self.root.bind("<r>", lambda _e: self.show_random())
        self.root.bind("<R>", lambda _e: self.show_random())
        self.root.bind("<Home>", lambda _e: self.show_first())
        self.root.bind("<End>", lambda _e: self.show_last())
        self.root.bind("<f>", lambda _e: self.fit_view())
        self.root.bind("<F>", lambda _e: self.fit_view())
        self.root.bind("<plus>", lambda _e: self.adjust_zoom(1.25))
        self.root.bind("<equal>", lambda _e: self.adjust_zoom(1.25))
        self.root.bind("<minus>", lambda _e: self.adjust_zoom(0.8))
        self.root.bind("<Key-0>", lambda _e: self.actual_size_view())

    def _sync_interaction_mode(self) -> None:
        tool = self.tool_mode.get()
        if tool == "ライン":
            self.main_canvas.set_interaction_mode("line")
        elif tool == "ROI":
            shape = self.roi_mode.get()
            if shape == "矩形":
                self.main_canvas.set_interaction_mode("rect")
            elif shape == "円":
                self.main_canvas.set_interaction_mode("circle")
            else:
                self.main_canvas.set_interaction_mode("freehand")
        else:
            self.main_canvas.set_interaction_mode("none")

    # ---------- File tree / navigation ----------
    def open_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="画像ファイルを選択",
            filetypes=[
                ("対応形式", "*.npy *.raw *.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("NumPy", "*.npy"),
                ("RAW", "*.raw"),
                ("画像", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("すべて", "*.*"),
            ],
        )
        if not paths:
            return
        self._load_paths([Path(p) for p in paths])

    def open_directory(self) -> None:
        path = filedialog.askdirectory(title="フォルダを選択")
        if not path:
            return
        root = Path(path)
        if not directory_has_images(root):
            messagebox.showwarning("フォルダ", "対応形式の画像が見つかりませんでした。")
            return
        self._load_paths([root])

    def add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="追加する画像ファイルを選択",
            filetypes=[("対応形式", "*.npy *.raw *.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("すべて", "*.*")],
        )
        if not paths:
            return
        new_paths = [Path(p) for p in paths if is_supported_image(Path(p))]
        if not new_paths:
            return
        for p in new_paths:
            if p not in self.root_sources:
                self.root_sources.append(p)
        self._rebuild_file_tree(preserve_path=self.current_path)
        if self.current_index < 0 and self.file_list:
            self.show_index(0)

    def _load_paths(self, paths: list[Path]) -> None:
        valid = [
            p
            for p in paths
            if p.exists()
            and ((p.is_file() and is_supported_image(p)) or (p.is_dir() and directory_has_images(p)))
        ]
        if not valid:
            messagebox.showwarning("読み込み", "有効なパスがありません。")
            return
        self.root_sources = valid
        self._rebuild_file_tree()
        if self.file_list:
            self.show_index(0)

    def _rebuild_file_tree(self, *, preserve_path: Path | None = None) -> None:
        self.file_list = []
        self._file_tree_iids = {}
        self.file_tree.delete(*self.file_tree.get_children(""))

        dir_counter = [0]
        for root in self.root_sources:
            if root.is_file() and is_supported_image(root):
                self._insert_file_node("", root)
            elif root.is_dir():
                node = build_dir_node(root)
                if node is not None:
                    self._insert_dir_node("", node, dir_counter)

        if preserve_path is not None and preserve_path in self.file_list:
            self.show_index(self.file_list.index(preserve_path))
        elif self.current_index >= len(self.file_list):
            if self.file_list:
                self.show_index(len(self.file_list) - 1, from_tree=True)

    def _insert_dir_node(self, parent: str, node: DirNode, dir_counter: list[int]) -> str:
        dir_counter[0] += 1
        label = f"{format_dir_number(dir_counter[0])}  {node.path.name}/"
        iid = self.file_tree.insert(parent, tk.END, text=label, tags=("dir",), open=True)
        for file_path in node.files:
            self._insert_file_node(iid, file_path)
        for child in node.children:
            self._insert_dir_node(iid, child, dir_counter)
        return iid

    def _insert_file_node(self, parent: str, path: Path) -> str:
        file_index = len(self.file_list)
        self.file_list.append(path)
        label = f"{format_file_number(file_index)}  {path.name}"
        iid = self.file_tree.insert(parent, tk.END, text=label, tags=("file",), values=(str(file_index),))
        self._file_tree_iids[file_index] = iid
        return iid

    def _file_index_from_iid(self, iid: str) -> int | None:
        if not iid:
            return None
        tags = self.file_tree.item(iid, "tags")
        if "file" not in tags:
            return None
        values = self.file_tree.item(iid, "values")
        if not values:
            return None
        return int(values[0])

    def _on_tree_select(self, _event: tk.Event) -> None:
        iid = self.file_tree.focus()
        index = self._file_index_from_iid(iid) if iid else None
        if index is not None:
            self.show_index(index, from_tree=True)

    def _on_tree_activate(self, _event: tk.Event) -> None:
        iid = self.file_tree.focus()
        index = self._file_index_from_iid(iid) if iid else None
        if index is not None:
            self.show_index(index)

    def show_prev(self) -> None:
        if not self.file_list:
            return
        if self.current_index <= 0:
            self.show_index(len(self.file_list) - 1)
        else:
            self.show_index(self.current_index - 1)

    def show_next(self) -> None:
        if not self.file_list:
            return
        mode = self.browse_mode.get()
        if mode == "ランダム":
            self.show_random()
            return
        if mode == "番号指定":
            self.jump_to_index()
            return
        if self.current_index >= len(self.file_list) - 1:
            self.show_index(0)
        else:
            self.show_index(self.current_index + 1)

    def show_random(self) -> None:
        if not self.file_list:
            return
        if len(self.file_list) == 1:
            self.show_index(0)
            return
        candidates = list(range(len(self.file_list)))
        if self.current_index in candidates:
            candidates.remove(self.current_index)
        self.show_index(random.choice(candidates))

    def show_first(self) -> None:
        if self.file_list:
            self.show_index(0)

    def show_last(self) -> None:
        if self.file_list:
            self.show_index(len(self.file_list) - 1)

    def jump_to_index(self) -> None:
        if not self.file_list:
            return
        text = self.index_var.get().strip()
        if not text.isdigit():
            messagebox.showwarning("番号", "番号には整数を入力してください。")
            return
        number = int(text)
        if number < 1 or number > len(self.file_list):
            messagebox.showwarning(
                "番号",
                f"1〜{format_file_number(len(self.file_list) - 1)} の番号を入力してください。",
            )
            return
        self.show_index(number - 1)

    def show_index(self, index: int, *, from_tree: bool = False) -> None:
        if not self.file_list or index < 0 or index >= len(self.file_list):
            return
        self.current_index = index
        path = self.file_list[index]
        self.index_var.set(format_file_number(index))
        if not from_tree:
            iid = self._file_tree_iids.get(index)
            if iid:
                self.file_tree.selection_set(iid)
                self.file_tree.focus(iid)
                self.file_tree.see(iid)
        self._load_current_image(path)

    def _load_current_image(self, path: Path) -> None:
        try:
            raw_settings = None
            if path.suffix.lower() == ".raw":
                guessed = guess_raw_settings(path, self.raw_settings)
                dialog = RawSettingsDialog(self.root, path, guessed)
                self.root.wait_window(dialog)
                if dialog.result is None:
                    return
                self.raw_settings = dict(dialog.result)
                raw_settings = self.raw_settings

            array, image, file_type = load_image_file(path, raw_settings)
            self.source_array = array
            self.base_image = image
            self.current_path = path
            self.current_file_type = file_type
            self.clear_roi(refresh=False)
            self.clear_line(refresh=False)
            self.compare_result = None
            self._rebuild_working_array()
            self.path_label.config(text=path.name)
            self.refresh_view(fit=True)
            self.update_stats_panel()
            self.update_histogram()
            self.update_quality()
            self.update_fft()
            self.status_var.set(f"読み込み完了: {path.name}")
        except Exception as exc:
            messagebox.showerror("読み込みエラー", str(exc))
            self.status_var.set("読み込み失敗")

    def reload_raw(self) -> None:
        if self.current_path is None or self.current_path.suffix.lower() != ".raw":
            messagebox.showinfo("RAW再読込", "現在のファイルは RAW ではありません。")
            return
        self._load_current_image(self.current_path)

    # ---------- View / display ----------
    def _resize_scale(self) -> float:
        return max(1, min(10, int(self.resize_var.get()))) / 10.0

    def _get_resize_method_name(self) -> str:
        name = self.resize_method_var.get()
        if name in RESIZE_METHODS:
            return name
        return DEFAULT_RESIZE_METHOD

    @staticmethod
    def _block_mean_2d(
        src: np.ndarray,
        y0: np.ndarray,
        y1: np.ndarray,
        x0: np.ndarray,
        x1: np.ndarray,
    ) -> np.ndarray:
        """各出力画素に対応する矩形ブロックの平均値を積分画像で計算する。"""
        h, w = src.shape
        padded = np.zeros((h + 1, w + 1), dtype=np.float64)
        padded[1:, 1:] = src
        integral = padded.cumsum(0).cumsum(1)

        iy0 = y0[:, None]
        iy1 = y1[:, None]
        ix0 = x0[None, :]
        ix1 = x1[None, :]
        sums = integral[iy1, ix1] - integral[iy0, ix1] - integral[iy1, ix0] + integral[iy0, ix0]
        area = (iy1 - iy0).astype(np.float64) * (ix1 - ix0).astype(np.float64)
        return sums / np.maximum(area, 1.0)

    @classmethod
    def _resize_array_block_mean(cls, array: np.ndarray, scale: float) -> np.ndarray:
        """配列を scale 倍に縮小（対応ブロックの周辺平均）。"""
        arr = np.asarray(array)
        if scale >= 0.999:
            return arr
        h, w = arr.shape[:2]
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        if nh == h and nw == w:
            return arr

        y0 = (np.arange(nh) * h) // nh
        y1 = ((np.arange(nh) + 1) * h) // nh
        x0 = (np.arange(nw) * w) // nw
        x1 = ((np.arange(nw) + 1) * w) // nw
        y1 = np.maximum(y1, y0 + 1)
        x1 = np.maximum(x1, x0 + 1)

        src = arr.astype(np.float64, copy=False)
        if arr.ndim == 2:
            out = cls._block_mean_2d(src, y0, y1, x0, x1)
        else:
            channels = [cls._block_mean_2d(src[..., c], y0, y1, x0, x1) for c in range(arr.shape[-1])]
            out = np.stack(channels, axis=-1)

        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            out = np.clip(np.rint(out), info.min, info.max).astype(arr.dtype)
        else:
            out = out.astype(arr.dtype, copy=False)
        return out

    @staticmethod
    def _resize_array_pil(array: np.ndarray, scale: float, resampling: Image.Resampling) -> np.ndarray:
        arr = np.asarray(array)
        if scale >= 0.999:
            return arr
        h, w = arr.shape[:2]
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        if nh == h and nw == w:
            return arr

        original_dtype = arr.dtype
        is_int = np.issubdtype(original_dtype, np.integer)

        def resize_plane(plane: np.ndarray) -> np.ndarray:
            # PIL の "F" モードは float32。float64 を渡すとバッファが誤読されノイズ化する。
            pil = Image.fromarray(np.ascontiguousarray(plane, dtype=np.float32), mode="F")
            return np.array(pil.resize((nw, nh), resampling), dtype=np.float64)

        if arr.ndim == 2:
            out = resize_plane(arr)
        else:
            out = np.stack([resize_plane(arr[..., c]) for c in range(arr.shape[-1])], axis=-1)

        if is_int:
            info = np.iinfo(original_dtype)
            out = np.clip(np.rint(out), info.min, info.max).astype(original_dtype)
        else:
            out = out.astype(original_dtype, copy=False)
        return out

    def _resize_array(self, array: np.ndarray, scale: float) -> np.ndarray:
        method_name = self._get_resize_method_name()
        if method_name == BLOCK_MEAN_RESIZE_METHOD:
            return self._resize_array_block_mean(array, scale)
        resampling = RESIZE_METHODS[method_name]
        return self._resize_array_pil(array, scale, resampling)

    def _rebuild_working_array(self) -> None:
        if self.source_array is None:
            self.working_array = None
            return
        self.working_array = self._resize_array(self.source_array, self._resize_scale())

    def _analysis_array(self) -> np.ndarray | None:
        """描画・対話解析に使う配列（リサイズ後）。"""
        return self.working_array if self.working_array is not None else self.source_array

    def _on_resize_slider(self, value: str) -> None:
        snapped = max(1, min(10, int(round(float(value)))))
        if snapped != int(self.resize_var.get()):
            self.resize_var.set(snapped)
        scale = snapped / 10.0
        if self.resize_value_label is not None:
            self.resize_value_label.config(text=f"{scale:.1f}x")
        self._schedule_resize()

    def _on_resize_method_change(self) -> None:
        self._schedule_resize()

    def _schedule_resize(self) -> None:
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(80, self._apply_resize)

    def _apply_resize(self) -> None:
        self._resize_job = None
        if self.source_array is None:
            return
        self.clear_roi(refresh=False)
        self.clear_line(refresh=False)
        self.compare_result = None
        self._rebuild_working_array()
        self.refresh_view(fit=True)
        self.update_stats_panel()
        self.update_histogram()
        self.update_quality()
        self.update_fft()
        if self.compare_array is not None:
            self.run_compare(silent=True)
        arr = self._analysis_array()
        if arr is not None:
            h, w = to_float_gray(arr).shape
            method = self._get_resize_method_name()
            self.status_var.set(f"リサイズ {self._resize_scale():.1f}x [{method}] → {w}×{h}")

    def fit_view(self) -> None:
        self.main_canvas.fit_to_window()

    def actual_size_view(self) -> None:
        self.main_canvas.actual_size()

    def adjust_zoom(self, factor: float) -> None:
        self.main_canvas.set_zoom(self.main_canvas.get_zoom() * factor)

    def refresh_view(self, *, fit: bool = False) -> None:
        arr = self._analysis_array()
        if arr is None:
            self.main_canvas.set_image(None)
            return

        mode = self.view_mode.get()
        cmap = self.cmap_var.get()

        if mode == "FFT":
            mag = compute_fft_magnitude(arr)
            display = render_fft_image(mag, cmap=cmap if cmap != "gray" else "inferno")
        elif mode == "差分ヒートマップ":
            if self.compare_result is None:
                if self.compare_array is not None:
                    self.run_compare(silent=True)
                if self.compare_result is None:
                    display = array_to_display_image(arr, cmap=None)
                    self.status_var.set("比較画像が未設定のため通常表示です")
                else:
                    display = abs_diff_heatmap(self.compare_result["abs_diff"], cmap="hot")
            else:
                display = abs_diff_heatmap(self.compare_result["abs_diff"], cmap="hot")
        elif mode == "カラーマップ":
            display = array_to_display_image(arr, cmap=cmap)
        else:
            display = array_to_display_image(arr, cmap=None)

        # 表示解像度を解析配列に合わせる
        gray = to_float_gray(arr)
        if display.size != (gray.shape[1], gray.shape[0]):
            display = display.resize((gray.shape[1], gray.shape[0]), Image.Resampling.NEAREST)

        display = overlay_roi_on_image(
            display,
            rect=self.roi_rect if self._drag_preview is None else None,
            circle=self.roi_circle if self._drag_preview is None else None,
            polygon=self.roi_polygon if self.roi_polygon and self._drag_preview is None else None,
            line=self.line_points if self._drag_preview is None else None,
        )
        if self._drag_preview is not None:
            kind = self._drag_preview[0]
            if kind == "rect":
                display = overlay_roi_on_image(display, rect=self._drag_preview[1])
            elif kind == "circle":
                display = overlay_roi_on_image(display, circle=self._drag_preview[1])
            elif kind == "freehand":
                display = overlay_roi_on_image(display, polygon=list(self._drag_preview[1]))
            elif kind == "line":
                display = overlay_roi_on_image(display, line=self._drag_preview[1])

        self.main_canvas.set_image(display, source_array=arr)
        if fit:
            self.root.update_idletasks()
            self.main_canvas.fit_to_window()

    def _on_main_cursor_move(self, x: int, y: int, _event: tk.Event | None) -> None:
        if x < 0 or self._analysis_array() is None or self.current_path is None:
            return
        val = self.main_canvas.get_pixel_value(x, y)
        self.status_var.set(f"{self.current_path.name}  ({x}, {y}) = {val}")

    # ---------- ROI / line interaction ----------
    def clear_roi(self, *, refresh: bool = True) -> None:
        self.roi_rect = None
        self.roi_circle = None
        self.roi_polygon = []
        self.roi_mask = None
        self._drag_preview = None
        self._set_text(self.roi_text, "ROI 未設定")
        if refresh:
            self.refresh_view()
            self.update_histogram()

    def clear_line(self, *, refresh: bool = True) -> None:
        self.line_points = None
        self._drag_preview = None
        self.profile_info.config(text="ライン未設定")
        self.profile_label.config(image="")
        self._profile_photo = None
        if refresh:
            self.refresh_view()

    def _on_canvas_click(self, x: int, y: int) -> None:
        if self.tool_mode.get() != "ROI" or self.roi_mode.get() != "自由選択":
            return
        self.roi_polygon.append((x, y))
        if len(self.roi_polygon) >= 3:
            self._apply_polygon_roi()
        self.refresh_view()

    def _on_canvas_drag(self, x0: int, y0: int, x1: int, y1: int, phase: str) -> None:
        mode = self.main_canvas.interaction_mode
        if mode == "rect":
            self._drag_preview = ("rect", (x0, y0, x1, y1))
        elif mode == "circle":
            r = float(np.hypot(x1 - x0, y1 - y0))
            self._drag_preview = ("circle", (x0, y0, r))
        elif mode == "line":
            self._drag_preview = ("line", (x0, y0, x1, y1))
        elif mode == "freehand":
            if phase == "start":
                self.roi_polygon = [(x0, y0)]
            self.roi_polygon.append((x1, y1))
            self._drag_preview = ("freehand", list(self.roi_polygon))
        self.refresh_view()

    def _on_canvas_drag_end(self, x0: int, y0: int, x1: int, y1: int) -> None:
        mode = self.main_canvas.interaction_mode
        self._drag_preview = None
        arr = self._analysis_array()
        if arr is None:
            return
        shape = to_float_gray(arr).shape

        if mode == "rect":
            self.roi_rect = (x0, y0, x1, y1)
            self.roi_circle = None
            self.roi_polygon = []
            self.roi_mask = rect_mask(shape, x0, y0, x1, y1)
            self.update_roi_stats()
            self.update_histogram()
        elif mode == "circle":
            r = float(np.hypot(x1 - x0, y1 - y0))
            self.roi_circle = (x0, y0, r)
            self.roi_rect = None
            self.roi_polygon = []
            self.roi_mask = circle_mask(shape, x0, y0, r)
            self.update_roi_stats()
            self.update_histogram()
        elif mode == "line":
            self.line_points = (x0, y0, x1, y1)
            self.update_profile()
        elif mode == "freehand":
            self.roi_polygon.append((x1, y1))
            self._apply_polygon_roi()
        self.refresh_view()

    def _apply_polygon_roi(self) -> None:
        arr = self._analysis_array()
        if arr is None or len(self.roi_polygon) < 3:
            return
        shape = to_float_gray(arr).shape
        self.roi_rect = None
        self.roi_circle = None
        self.roi_mask = polygon_mask(shape, self.roi_polygon)
        self.update_roi_stats()
        self.update_histogram()

    # ---------- Analysis updates ----------
    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.config(state=tk.DISABLED)

    @staticmethod
    def _format_stats(st: dict, title: str = "統計") -> str:
        lines = [
            f"[{title}]",
            f"shape   : {st['shape']}",
            f"dtype   : {st['dtype']}",
            f"count   : {st['count']}",
            f"min     : {st['min']:.6g}",
            f"max     : {st['max']:.6g}",
            f"mean    : {st['mean']:.6g}",
            f"std     : {st['std']:.6g}",
            f"median  : {st['median']:.6g}",
            f"p01/p05 : {st['p01']:.6g} / {st['p05']:.6g}",
            f"p25/p75 : {st['p25']:.6g} / {st['p75']:.6g}",
            f"p95/p99 : {st['p95']:.6g} / {st['p99']:.6g}",
            f"zero%   : {st['zero_ratio'] * 100:.2f}",
            f"satur%  : {st['saturated_ratio'] * 100:.2f}",
            f"NaN/Inf : {st['nan_count']} / {st['inf_count']}",
        ]
        return "\n".join(lines)

    def update_stats_panel(self) -> None:
        arr = self._analysis_array()
        if arr is None or self.source_array is None:
            self._set_text(self.meta_text, "")
            return
        st = compute_stats(arr)
        scale = self._resize_scale()
        src_h, src_w = to_float_gray(self.source_array).shape
        work_h, work_w = to_float_gray(arr).shape
        header = []
        if self.current_path is not None:
            header.append(f"file: {self.current_path.name}")
        header.append(f"resize : {scale:.1f}x ({self._get_resize_method_name()})")
        header.append(f"src    : {src_w}×{src_h}")
        header.append(f"work   : {work_w}×{work_h}")
        text = "\n".join(header) + "\n\n" + self._format_stats(st, "解析対象（リサイズ後）")
        self._set_text(self.meta_text, text)

    def update_histogram(self) -> None:
        arr = self._analysis_array()
        if arr is None:
            return
        hist = compute_histogram(
            arr,
            bins=max(16, int(self.hist_bins.get())),
            mask=self.roi_mask,
            channel=self.hist_channel.get(),
        )
        img = render_histogram_image(hist, show_cdf=self.show_cdf.get())
        self._plot_photo = ImageTk.PhotoImage(img)
        self.hist_label.config(image=self._plot_photo)

    def update_roi_stats(self) -> None:
        arr = self._analysis_array()
        if arr is None or self.roi_mask is None or not self.roi_mask.any():
            self._set_text(self.roi_text, "有効な ROI がありません。")
            return
        st = compute_stats(arr, mask=self.roi_mask)
        info = []
        if self.roi_rect is not None:
            info.append(f"矩形: {self.roi_rect}")
        if self.roi_circle is not None:
            info.append(f"円: center=({self.roi_circle[0]}, {self.roi_circle[1]}) r={self.roi_circle[2]:.1f}")
        if self.roi_polygon:
            info.append(f"多角形: 頂点数={len(self.roi_polygon)}")
        text = "\n".join(info) + "\n\n" + self._format_stats(st, "ROI")
        self._set_text(self.roi_text, text)

    def update_profile(self) -> None:
        arr = self._analysis_array()
        if arr is None or self.line_points is None:
            return
        x0, y0, x1, y1 = self.line_points
        profile = extract_line_profile(arr, x0, y0, x1, y1)
        img = render_profile_image(profile["distance"], profile["values"])
        self._profile_photo = ImageTk.PhotoImage(img)
        self.profile_label.config(image=self._profile_photo)
        vals = profile["values"]
        self.profile_info.config(
            text=(
                f"({x0},{y0})→({x1},{y1})  "
                f"len={profile['distance'][-1]:.1f}px  "
                f"min={vals.min():.4g} max={vals.max():.4g} mean={vals.mean():.4g}"
            )
        )

    def update_quality(self) -> None:
        arr = self._analysis_array()
        if arr is None:
            return
        q = assess_quality(arr)
        lines = [
            "[品質チェック]",
            f"blur_score (Laplacian var): {q.blur_score:.2f}  {'⚠' if q.blur_flag else 'OK'}",
            f"underexposed ratio        : {q.underexposed_ratio * 100:.2f}%  {'⚠' if q.underexposed_flag else 'OK'}",
            f"overexposed ratio         : {q.overexposed_ratio * 100:.2f}%  {'⚠' if q.overexposed_flag else 'OK'}",
            f"noise estimate            : {q.noise_estimate:.2f}  {'⚠' if q.noise_flag else 'OK'}",
            f"resolution OK             : {q.resolution_ok}",
            "",
            "[メモ]",
            *q.notes,
        ]
        self._set_text(self.quality_text, "\n".join(lines))

    def update_fft(self) -> None:
        arr = self._analysis_array()
        if arr is None:
            return
        mag = compute_fft_magnitude(arr)
        # サムネイル表示用に縮小
        img = render_fft_image(mag, cmap=self.cmap_var.get() if self.cmap_var.get() != "gray" else "inferno")
        thumb = img.copy()
        thumb.thumbnail((400, 400), Image.Resampling.BILINEAR)
        self._fft_photo = ImageTk.PhotoImage(thumb)
        self.fft_label.config(image=self._fft_photo)

    def open_compare_image(self) -> None:
        path = filedialog.askopenfilename(
            title="比較する画像を選択",
            filetypes=[("対応形式", "*.npy *.raw *.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("すべて", "*.*")],
        )
        if not path:
            return
        p = Path(path)
        try:
            raw_settings = None
            if p.suffix.lower() == ".raw":
                guessed = guess_raw_settings(p, self.raw_settings)
                dialog = RawSettingsDialog(self.root, p, guessed)
                self.root.wait_window(dialog)
                if dialog.result is None:
                    return
                raw_settings = dict(dialog.result)
            array, _image, _ftype = load_image_file(p, raw_settings)
            self.compare_array = array
            self.compare_path = p
            self.compare_path_label.config(text=f"比較画像: {p.name}")
            self.run_compare()
        except Exception as exc:
            messagebox.showerror("比較画像", str(exc))

    def run_compare(self, *, silent: bool = False) -> None:
        arr = self._analysis_array()
        if arr is None or self.compare_array is None:
            if not silent:
                messagebox.showinfo("比較", "現在画像と比較画像の両方が必要です。")
            return
        # 比較画像も同じリサイズ率で揃える
        compare_resized = self._resize_array(self.compare_array, self._resize_scale())
        result = compare_images(arr, compare_resized)
        self.compare_result = result
        psnr_val = result["psnr"]
        psnr_text = "inf" if np.isinf(psnr_val) else f"{psnr_val:.4f}"
        text = "\n".join(
            [
                f"比較先: {self.compare_path.name if self.compare_path else '-'}",
                f"比較サイズ: {result['shape']}",
                f"リサイズ: {self._resize_scale():.1f}x",
                f"MSE  : {result['mse']:.6g}",
                f"MAE  : {result['mae']:.6g}",
                f"PSNR : {psnr_text} dB",
                f"SSIM : {result['ssim']:.6f}",
            ]
        )
        self._set_text(self.compare_text, text)
        heat = abs_diff_heatmap(result["abs_diff"], cmap="hot")
        heat.thumbnail((400, 320), Image.Resampling.BILINEAR)
        self._compare_photo = ImageTk.PhotoImage(heat)
        self.compare_label.config(image=self._compare_photo)
        if self.view_mode.get() == "差分ヒートマップ":
            self.refresh_view()

    # ---------- Export ----------
    def choose_export_dir(self) -> None:
        path = filedialog.askdirectory(title="保存先フォルダを選択", initialdir=str(self.export_dir))
        if path:
            self.export_dir = Path(path)
            self.status_var.set(f"保存先: {self.export_dir}")

    def _unique_path(self, directory: Path, stem: str, suffix: str) -> Path:
        candidate = directory / f"{stem}{suffix}"
        if not candidate.exists():
            return candidate
        i = 1
        while True:
            candidate = directory / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                return candidate
            i += 1

    def save_display_image(self) -> None:
        img = self.main_canvas.get_image()
        if img is None:
            return
        stem = (self.current_path.stem if self.current_path else "display") + "_analyzed"
        path = filedialog.asksaveasfilename(
            title="解析画像を保存",
            initialdir=str(self.export_dir),
            initialfile=f"{stem}.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if not path:
            return
        img.save(path)
        self.status_var.set(f"保存しました: {path}")

    def save_histogram_image(self) -> None:
        arr = self._analysis_array()
        if self._plot_photo is None or arr is None:
            return
        hist = compute_histogram(
            arr,
            bins=max(16, int(self.hist_bins.get())),
            mask=self.roi_mask,
            channel=self.hist_channel.get(),
        )
        img = render_histogram_image(hist, show_cdf=self.show_cdf.get())
        stem = (self.current_path.stem if self.current_path else "hist") + "_histogram"
        out = self._unique_path(self.export_dir, stem, ".png")
        img.save(out)
        self.status_var.set(f"保存しました: {out}")

    def save_profile_image(self) -> None:
        arr = self._analysis_array()
        if arr is None or self.line_points is None:
            messagebox.showinfo("プロファイル", "ラインが未設定です。")
            return
        x0, y0, x1, y1 = self.line_points
        profile = extract_line_profile(arr, x0, y0, x1, y1)
        img = render_profile_image(profile["distance"], profile["values"])
        stem = (self.current_path.stem if self.current_path else "profile") + "_profile"
        out = self._unique_path(self.export_dir, stem, ".png")
        img.save(out)
        self.status_var.set(f"保存しました: {out}")

    def save_fft_image(self) -> None:
        arr = self._analysis_array()
        if arr is None:
            return
        mag = compute_fft_magnitude(arr)
        img = render_fft_image(mag, cmap=self.cmap_var.get() if self.cmap_var.get() != "gray" else "inferno")
        stem = (self.current_path.stem if self.current_path else "fft") + "_fft"
        out = self._unique_path(self.export_dir, stem, ".png")
        img.save(out)
        self.status_var.set(f"保存しました: {out}")

    def run_batch_export(self, fmt: str) -> None:
        if not self.file_list:
            messagebox.showinfo("バッチ", "ファイル一覧が空です。")
            return
        if self._batch_busy:
            messagebox.showinfo("バッチ", "処理中です。")
            return

        if fmt == "csv":
            path = filedialog.asksaveasfilename(
                title="CSVの保存先",
                initialdir=str(self.export_dir),
                initialfile="batch_analysis.csv",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")],
            )
        else:
            path = filedialog.asksaveasfilename(
                title="JSONの保存先",
                initialdir=str(self.export_dir),
                initialfile="batch_analysis.json",
                defaultextension=".json",
                filetypes=[("JSON", "*.json")],
            )
        if not path:
            return

        include_quality = self.batch_include_quality.get()
        targets = list(self.file_list)
        raw_settings = dict(self.raw_settings)

        def worker() -> None:
            rows: list[dict] = []
            errors: list[str] = []
            for i, fp in enumerate(targets):
                self.root.after(
                    0,
                    lambda i=i, n=len(targets), name=fp.name: self.batch_status.config(
                        text=f"処理中... {i + 1}/{n}  {name}"
                    ),
                )
                try:
                    settings = raw_settings if fp.suffix.lower() == ".raw" else None
                    if settings is not None:
                        settings = guess_raw_settings(fp, raw_settings)
                    array, _img, _t = load_image_file(fp, settings)
                    quality = assess_quality(array) if include_quality else None
                    rows.append(stats_row_for_export(fp, array, quality))
                except Exception as exc:
                    errors.append(f"{fp.name}: {exc}")

            try:
                out = Path(path)
                if fmt == "csv":
                    if rows:
                        fieldnames = list(rows[0].keys())
                        with out.open("w", newline="", encoding="utf-8-sig") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)
                else:
                    with out.open("w", encoding="utf-8") as f:
                        json.dump({"rows": rows, "errors": errors}, f, ensure_ascii=False, indent=2)
                msg = f"完了: {len(rows)} 件 → {out}"
                if errors:
                    msg += f"（失敗 {len(errors)} 件）"
                self.root.after(0, lambda: self._batch_done(msg, errors))
            except Exception as exc:
                self.root.after(0, lambda: self._batch_done(f"保存失敗: {exc}", errors))

        self._batch_busy = True
        self.batch_status.config(text="バッチ処理を開始しています...")
        threading.Thread(target=worker, daemon=True).start()

    def _batch_done(self, message: str, errors: list[str]) -> None:
        self._batch_busy = False
        self.batch_status.config(text=message)
        self.status_var.set(message)
        if errors:
            messagebox.showwarning("バッチ", message + "\n\n" + "\n".join(errors[:10]))
        else:
            messagebox.showinfo("バッチ", message)

    def show_shortcuts_help(self) -> None:
        messagebox.showinfo(
            "ショートカット",
            "\n".join(
                [
                    "Ctrl+O : ファイルを開く",
                    "Ctrl+Shift+O : フォルダを開く",
                    "← / → : 前 / 次",
                    "R : ランダム",
                    "Home / End : 先頭 / 末尾",
                    "F : Fit",
                    "+ / - : 拡大 / 縮小",
                    "0 : 100%",
                    "中クリック・右ドラッグ / Ctrl+左 : パン",
                ]
            ),
        )


def main() -> None:
    initial: list[Path] = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.exists():
            initial.append(p)

    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    ImageAnalyzerApp(root, initial_paths=initial or None)
    root.mainloop()


if __name__ == "__main__":
    main()
