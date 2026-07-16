"""ディレクトリ／単体ファイル対応の多機能画像ビューワー。"""

from __future__ import annotations

import random
import sys
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageOps, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    array_stats,
    guess_raw_settings,
    is_supported_image,
    load_image_file,
)
from module.image_processing import (  # noqa: E402
    DEFAULT_RESIZE_METHOD,
    RESIZE_METHODS,
    apply_processing,
    default_params,
)


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
        value = int(text)
        if value < 0:
            raise ValueError(f"{name}には0以上の整数を入力してください。")
        return value

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


class MagnifierWindow(tk.Toplevel):
    """マウスホバー位置を拡大表示するルーペウィンドウ。"""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.withdraw()
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.label = ttk.Label(self, relief=tk.SOLID, borderwidth=1)
        self.label.pack()
        self._photo: ImageTk.PhotoImage | None = None

    def show_patch(self, patch: Image.Image, screen_x: int, screen_y: int) -> None:
        self._photo = ImageTk.PhotoImage(patch)
        self.label.config(image=self._photo)
        self.geometry(f"+{screen_x + 18}+{screen_y + 18}")
        self.deiconify()

    def hide(self) -> None:
        self.withdraw()


class ZoomableCanvas(ttk.Frame):
    """ズーム・パン対応の画像キャンバス。"""

    PAN_SPEED = 0.5

    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_cursor_move: Callable[..., None] | None = None,
        on_click: Callable[..., None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.on_cursor_move = on_cursor_move
        self.on_click = on_click

        self.canvas = tk.Canvas(self, background="#1e1e1e", highlightthickness=0)
        h_scroll = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self._on_hscroll)
        v_scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._on_vscroll)
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
        self._resampling: Image.Resampling = Image.Resampling.LANCZOS
        self._pan_start: tuple[int, int] | None = None
        self._pan_remainder_x = 0.0
        self._pan_remainder_y = 0.0
        self._scroll_region = (0, 0, 1, 1)

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
        self.canvas.bind("<ButtonPress-1>", self._on_left_click)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())

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

    def set_resampling(self, resampling: Image.Resampling) -> None:
        if resampling != self._resampling:
            self._resampling = resampling
            self._redraw()

    def _on_hscroll(self, *args: str) -> None:
        self.canvas.xview(*args)
        self._redraw()

    def _on_vscroll(self, *args: str) -> None:
        self.canvas.yview(*args)
        self._redraw()

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
        self._redraw()

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
                if vals.ndim == 0:
                    return f"{vals}"
                return ", ".join(str(v) for v in np.asarray(vals).ravel()[:4])
        if self._image is None:
            return "-"
        px = self._image.getpixel((x, y))
        if isinstance(px, int):
            return str(px)
        return ", ".join(str(v) for v in px)

    def extract_patch(self, x: int, y: int, half_size: int, zoom_factor: int = 8) -> Image.Image | None:
        if self._image is None:
            return None
        left = max(0, x - half_size)
        top = max(0, y - half_size)
        right = min(self._image.width, x + half_size + 1)
        bottom = min(self._image.height, y + half_size + 1)
        patch = self._image.crop((left, top, right, bottom))
        target = half_size * 2 + 1
        display_size = max(target * zoom_factor, target)
        return patch.resize((display_size, display_size), Image.Resampling.NEAREST)

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
            self.canvas.config(scrollregion=(0, 0, 1, 1))
            return

        zoom = self._zoom
        img_w, img_h = self._image.width, self._image.height
        disp_w = max(1, int(round(img_w * zoom)))
        disp_h = max(1, int(round(img_h * zoom)))
        self.canvas.config(scrollregion=(0, 0, disp_w, disp_h))

        # 表示中の領域（ズーム後座標）だけを描画してメモリ消費を抑える。
        view_w = max(1, self.canvas.winfo_width())
        view_h = max(1, self.canvas.winfo_height())
        left = max(0.0, self.canvas.canvasx(0))
        top = max(0.0, self.canvas.canvasy(0))
        right = min(float(disp_w), left + view_w)
        bottom = min(float(disp_h), top + view_h)
        if right <= left or bottom <= top:
            return

        # ズーム後座標 → 元画像座標へ変換し、可視範囲を切り出す。
        src_left = max(0, min(img_w - 1, int(left / zoom)))
        src_top = max(0, min(img_h - 1, int(top / zoom)))
        src_right = max(src_left + 1, min(img_w, int(right / zoom) + 1))
        src_bottom = max(src_top + 1, min(img_h, int(bottom / zoom) + 1))

        region = self._image.crop((src_left, src_top, src_right, src_bottom))
        region_w = max(1, int(round((src_right - src_left) * zoom)))
        region_h = max(1, int(round((src_bottom - src_top) * zoom)))
        if (region.width, region.height) != (region_w, region_h):
            region = region.resize((region_w, region_h), self._resampling)

        self._photo = ImageTk.PhotoImage(region)
        self.canvas.create_image(
            int(round(src_left * zoom)),
            int(round(src_top * zoom)),
            image=self._photo,
            anchor=tk.NW,
        )

    def _on_mousewheel(self, event: tk.Event) -> None:
        if self._image is None:
            return
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        old_zoom = self._zoom
        new_zoom = max(0.05, min(32.0, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        # カーソル下の画像座標がズーム後も同じ位置に留まるようにする。
        img_x = self.canvas.canvasx(event.x) / old_zoom
        img_y = self.canvas.canvasy(event.y) / old_zoom
        self._zoom = new_zoom

        disp_w = max(1, int(round(self._image.width * new_zoom)))
        disp_h = max(1, int(round(self._image.height * new_zoom)))
        self.canvas.config(scrollregion=(0, 0, disp_w, disp_h))
        new_left = img_x * new_zoom - event.x
        new_top = img_y * new_zoom - event.y
        self.canvas.xview_moveto(max(0.0, new_left / disp_w))
        self.canvas.yview_moveto(max(0.0, new_top / disp_h))
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
        if scroll_x or scroll_y:
            self._redraw()

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

    def _on_left_click(self, event: tk.Event) -> None:
        if self.on_click is None or self._image is None:
            return
        pt = self.canvas_to_image(event.x, event.y)
        if pt is not None:
            self.on_click(pt[0], pt[1])


class ImageViewerApp:
    """多機能画像ビューワーのメインアプリケーション。"""

    BROWSE_MODES = ("順番", "ランダム", "番号指定")

    @staticmethod
    def format_list_number(index: int) -> str:
        return format_file_number(index)

    def __init__(self, root: tk.Tk, initial_paths: list[Path] | None = None) -> None:
        self.root = root
        self.root.title("画像ビューワー")
        self.root.minsize(1280, 820)

        self.file_list: list[Path] = []
        self.root_sources: list[Path] = []
        self._file_tree_iids: dict[int, str] = {}
        self.current_index = -1
        self.source_array: np.ndarray | None = None
        self.base_image: Image.Image | None = None
        self.current_path: Path | None = None
        self.current_file_type: str | None = None
        self.raw_settings: dict[str, object] = dict(DEFAULT_RAW_SETTINGS)

        self.export_dir = default_downloads_dir()

        self.param_values = default_params()
        self.param_vars: dict[str, tk.IntVar] = {}
        self.param_labels: dict[str, ttk.Label] = {}
        self.resize_method_var = tk.StringVar(value=DEFAULT_RESIZE_METHOD)
        self._update_job: str | None = None

        self.magnifier_enabled = tk.BooleanVar(value=True)
        self.magnifier_size = tk.IntVar(value=64)
        self.magnifier_zoom = tk.IntVar(value=2)
        self.magnifier_radius_label: ttk.Label | None = None
        self.magnifier_zoom_label: ttk.Label | None = None
        self.export_dir_label: ttk.Label | None = None
        self.magnifier = MagnifierWindow(self.root)

        self.browse_mode = tk.StringVar(value="順番")
        self.index_var = tk.StringVar(value="0001")

        self._build_ui()
        self._bind_shortcuts()

        if initial_paths:
            self._load_paths(initial_paths)

    def _build_ui(self) -> None:
        self._build_menu()
        self._build_toolbar()

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        left = ttk.Frame(main, padding=4, width=260)
        main.add(left, weight=0)
        center = ttk.Frame(main, padding=4)
        main.add(center, weight=3)
        right = ttk.Frame(main, padding=4, width=300)
        main.add(right, weight=0)

        self._build_file_panel(left)
        self._build_view_panel(center)
        self._build_param_panel(right)

        self.status_var = tk.StringVar(value="準備完了")
        status = ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W, padding=(8, 4))
        status.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ファイル", menu=file_menu)
        file_menu.add_command(label="ファイルを開く...", command=self.open_files, accelerator="Ctrl+O")
        file_menu.add_command(label="フォルダを開く...", command=self.open_directory, accelerator="Ctrl+Shift+O")
        file_menu.add_command(label="ファイルを追加...", command=self.add_files)
        file_menu.add_separator()
        export_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="エクスポート", menu=export_menu)
        export_menu.add_command(label="PNG", command=lambda: self.export_processed("png"))
        export_menu.add_command(label="JPEG", command=lambda: self.export_processed("jpeg"))
        export_menu.add_command(label="NPY", command=lambda: self.export_processed("npy"))
        export_menu.add_separator()
        export_menu.add_command(label="名前を付けて保存...", command=self.export_processed_as)
        export_menu.add_command(label="保存先フォルダを変更...", command=self.choose_export_dir)
        file_menu.add_separator()
        file_menu.add_command(label="終了", command=self.root.quit)

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="表示", menu=view_menu)
        view_menu.add_command(label="ウィンドウに合わせる", command=self.fit_view, accelerator="F")
        view_menu.add_command(label="実サイズ (100%)", command=self.actual_size_view, accelerator="0")
        view_menu.add_command(label="拡大", command=lambda: self.adjust_zoom(1.25), accelerator="+")
        view_menu.add_command(label="縮小", command=lambda: self.adjust_zoom(0.8), accelerator="-")
        view_menu.add_separator()
        view_menu.add_checkbutton(label="ルーペを表示", variable=self.magnifier_enabled)

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
        ttk.Combobox(
            bar,
            textvariable=self.browse_mode,
            values=self.BROWSE_MODES,
            state="readonly",
            width=12,
        ).pack(side=tk.LEFT)
        ttk.Label(bar, text="番号").pack(side=tk.LEFT, padx=(6, 2))
        index_entry = ttk.Entry(bar, textvariable=self.index_var, width=6)
        index_entry.pack(side=tk.LEFT)
        index_entry.bind("<Return>", lambda _e: self.jump_to_index())
        ttk.Button(bar, text="移動", command=self.jump_to_index).pack(side=tk.LEFT, padx=(2, 0))

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(bar, text="Fit", command=self.fit_view).pack(side=tk.LEFT)
        ttk.Button(bar, text="100%", command=self.actual_size_view).pack(side=tk.LEFT, padx=(2, 0))

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(bar, text="エクスポート:", font=("", 9, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(bar, text="PNG", command=lambda: self.export_processed("png")).pack(side=tk.LEFT)
        ttk.Button(bar, text="JPEG", command=lambda: self.export_processed("jpeg")).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Button(bar, text="NPY", command=lambda: self.export_processed("npy")).pack(side=tk.LEFT, padx=(2, 0))

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

        self.file_tree = ttk.Treeview(
            list_frame,
            show="tree",
            selectmode="browse",
            style="File.Treeview",
        )
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

        ttk.Label(parent, text="メタ情報").pack(anchor=tk.W, pady=(4, 0))
        self.meta_text = tk.Text(parent, height=10, width=30, state=tk.DISABLED, font=("Consolas", 9))
        self.meta_text.pack(fill=tk.BOTH, expand=False, pady=(2, 0))

    def _build_view_panel(self, parent: ttk.Frame) -> None:
        view_frame = ttk.LabelFrame(parent, text="プレビュー", padding=2)
        view_frame.pack(fill=tk.BOTH, expand=True)

        self.main_canvas = ZoomableCanvas(view_frame, on_cursor_move=self._on_main_cursor_move)
        self.main_canvas.pack(fill=tk.BOTH, expand=True)

        flip_row = ttk.Frame(parent)
        flip_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(flip_row, text="左右反転", command=self.flip_horizontal).pack(side=tk.LEFT)
        ttk.Button(flip_row, text="上下反転", command=self.flip_vertical).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(flip_row, text="90°回転", command=lambda: self.rotate_by(90)).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(flip_row, text="-90°回転", command=lambda: self.rotate_by(-90)).pack(side=tk.LEFT, padx=(4, 0))

    def _build_param_panel(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent)
        header.pack(fill=tk.X)
        ttk.Label(header, text="画像パラメータ", font=("", 10, "bold")).pack(side=tk.LEFT)
        ttk.Button(header, text="リセット", command=self.reset_params).pack(side=tk.RIGHT)

        scroll_container = ttk.Frame(parent)
        scroll_container.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        canvas = tk.Canvas(scroll_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_container, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        slider_defs = [
            ("resize", "リサイズ", 1, 10, 10),
            ("brightness", "明るさ", 0, 300, 100),
            ("contrast", "コントラスト", 0, 300, 100),
            ("gamma", "ガンマ", 10, 300, 100),
            ("clip_min", "クリップ下限", 0, 100, 0),
            ("clip_max", "クリップ上限", 0, 100, 100),
            ("blur", "ぼかし", 0, 10, 0),
            ("sharpen", "シャープ", 0, 100, 0),
            ("threshold", "二値化", 0, 255, 0),
            ("equalize", "均等化", 0, 100, 0),
            ("invert", "反転", 0, 100, 0),
            ("rotate", "回転 (°)", -180, 180, 0),
        ]
        for key, label, vmin, vmax, default in slider_defs:
            self._add_slider(inner, key, label, vmin, vmax, default)
            if key == "resize":
                self._add_resize_method_row(inner)

        mag_frame = ttk.LabelFrame(parent, text="ルーペ", padding=6)
        mag_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Checkbutton(mag_frame, text="有効", variable=self.magnifier_enabled).pack(anchor=tk.W)

        radius_row = ttk.Frame(mag_frame)
        radius_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(radius_row, text="取得半径 (px)").pack(side=tk.LEFT)
        self.magnifier_radius_label = ttk.Label(radius_row, text=str(self.magnifier_size.get()), width=4)
        self.magnifier_radius_label.pack(side=tk.RIGHT)
        ttk.Scale(
            mag_frame,
            from_=4,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.magnifier_size,
            command=self._on_magnifier_radius_change,
        ).pack(fill=tk.X, pady=(2, 0))

        zoom_row = ttk.Frame(mag_frame)
        zoom_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(zoom_row, text="拡大率").pack(side=tk.LEFT)
        self.magnifier_zoom_label = ttk.Label(zoom_row, text=f"{self.magnifier_zoom.get()}x", width=5)
        self.magnifier_zoom_label.pack(side=tk.RIGHT)
        ttk.Scale(
            mag_frame,
            from_=1,
            to=32,
            orient=tk.HORIZONTAL,
            variable=self.magnifier_zoom,
            command=self._on_magnifier_zoom_change,
        ).pack(fill=tk.X, pady=(2, 0))

        ttk.Label(parent, text="エクスポート", font=("", 10, "bold")).pack(anchor=tk.W, pady=(8, 0))
        export_frame = ttk.Frame(parent, padding=(0, 4))
        export_frame.pack(fill=tk.X)

        dir_row = ttk.Frame(export_frame)
        dir_row.pack(fill=tk.X)
        ttk.Label(dir_row, text="保存先").pack(side=tk.LEFT)
        ttk.Button(dir_row, text="変更", command=self.choose_export_dir).pack(side=tk.RIGHT)
        self.export_dir_label = ttk.Label(
            export_frame,
            text=self._format_export_dir(),
            wraplength=260,
            foreground="#555555",
        )
        self.export_dir_label.pack(anchor=tk.W, pady=(4, 6))

        export_buttons = ttk.Frame(export_frame)
        export_buttons.pack(fill=tk.X)
        ttk.Button(export_buttons, text="PNG", command=lambda: self.export_processed("png")).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3)
        )
        ttk.Button(export_buttons, text="JPEG", command=lambda: self.export_processed("jpeg")).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 3)
        )
        ttk.Button(export_buttons, text="NPY", command=lambda: self.export_processed("npy")).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0)
        )
        ttk.Button(export_frame, text="名前を付けて保存...", command=self.export_processed_as).pack(
            fill=tk.X, pady=(6, 0)
        )

    def _add_slider(
        self,
        parent: ttk.Frame,
        key: str,
        label: str,
        vmin: int,
        vmax: int,
        default: int,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
        var = tk.IntVar(value=default)
        self.param_vars[key] = var
        value_label = ttk.Label(row, text=self._format_param_value(key, default), width=5)
        value_label.pack(side=tk.RIGHT)
        self.param_labels[key] = value_label

        def on_change(val: str, k: str = key) -> None:
            value = int(float(val))
            if k == "resize":
                value = max(1, min(10, value))
            self.param_values[k] = value
            self.param_labels[k].config(text=self._format_param_value(k, value))
            self.schedule_refresh()

        scale = ttk.Scale(row, from_=vmin, to=vmax, orient=tk.HORIZONTAL, variable=var, command=on_change)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))

    def _add_resize_method_row(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(row, text="手法", width=12).pack(side=tk.LEFT)
        combo = ttk.Combobox(
            row,
            textvariable=self.resize_method_var,
            values=list(RESIZE_METHODS.keys()),
            state="readonly",
            width=22,
        )
        combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))
        combo.bind("<<ComboboxSelected>>", lambda _e: self.schedule_refresh())

    def _get_resize_resampling(self) -> Image.Resampling:
        name = self.resize_method_var.get()
        return RESIZE_METHODS.get(name, RESIZE_METHODS[DEFAULT_RESIZE_METHOD])

    @staticmethod
    def _format_param_value(key: str, value: int) -> str:
        if key == "resize":
            return f"{value / 10:.1f}x"
        return str(value)

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
        self.root.bind("<Escape>", lambda _e: self.magnifier.hide())

    def schedule_refresh(self) -> None:
        if self._update_job is not None:
            self.root.after_cancel(self._update_job)
        self._update_job = self.root.after(60, self._run_refresh)

    def _run_refresh(self) -> None:
        self._update_job = None
        self.refresh_view()

    def open_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="画像ファイルを選択",
            filetypes=[
                ("対応形式", "*.npy *.raw *.png *.jpg *.jpeg"),
                ("NumPy", "*.npy"),
                ("RAW", "*.raw"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
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
            filetypes=[("対応形式", "*.npy *.raw *.png *.jpg *.jpeg"), ("すべて", "*.*")],
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
        label = f"{self.format_list_number(file_index)}  {path.name}"
        iid = self.file_tree.insert(
            parent,
            tk.END,
            text=label,
            tags=("file",),
            values=(str(file_index),),
        )
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
                f"1〜{self.format_list_number(len(self.file_list) - 1)} の番号を入力してください。",
            )
            return
        self.show_index(number - 1)

    def show_index(self, index: int, *, from_tree: bool = False) -> None:
        if not self.file_list or index < 0 or index >= len(self.file_list):
            return
        self.current_index = index
        if not from_tree:
            iid = self._file_tree_iids.get(index)
            if iid:
                self.file_tree.selection_set(iid)
                self.file_tree.focus(iid)
                self.file_tree.see(iid)
        self.index_var.set(self.format_list_number(index))
        self.load_file(self.file_list[index])

    def load_file(self, path: Path) -> None:
        try:
            suffix = path.suffix.lower()
            if suffix == ".raw" or (suffix not in (".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
                settings = self._ask_raw_settings(path)
                if settings is None and suffix not in (".npy", ".png", ".jpg", ".jpeg"):
                    return
                if settings is not None:
                    self.raw_settings = settings
                array, image, file_type = load_image_file(path, self.raw_settings)
            else:
                array, image, file_type = load_image_file(path)
        except Exception as exc:
            messagebox.showerror("読み込みエラー", f"{path.name}\n{exc}")
            return

        self.current_path = path
        self.source_array = array
        self.base_image = image
        self.current_file_type = file_type
        self.path_label.config(text=str(path))
        self._update_meta(path, array)
        self.refresh_view()
        self.root.after(50, self.fit_view)

    def _ask_raw_settings(self, path: Path) -> dict[str, object] | None:
        suffix = path.suffix.lower()
        if suffix in (".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
            return None
        guessed = guess_raw_settings(path, self.raw_settings)
        dialog = RawSettingsDialog(self.root, path, guessed)
        self.root.wait_window(dialog)
        return dialog.result

    def reload_raw(self) -> None:
        if self.current_path is None or self.current_file_type != "raw":
            messagebox.showinfo("RAW", "RAWファイルを開いてください。")
            return
        settings = self._ask_raw_settings(self.current_path)
        if settings is None:
            return
        try:
            self.raw_settings = settings
            array, image, _ = load_image_file(self.current_path, settings)
        except Exception as exc:
            messagebox.showerror("読み込みエラー", str(exc))
            return
        self.source_array = array
        self.base_image = image
        self._update_meta(self.current_path, array)
        self.refresh_view()

    def _get_processed_image(self) -> Image.Image | None:
        if self.base_image is None:
            return None
        return apply_processing(
            self.base_image,
            self.param_values,
            resize_method=self._get_resize_resampling(),
        )

    def refresh_view(self) -> None:
        processed = self._get_processed_image()
        if processed is None:
            self.main_canvas.set_image(None)
            return
        self.main_canvas.set_resampling(self._get_resize_resampling())
        self.main_canvas.set_image(processed, self.source_array)

    def reset_params(self) -> None:
        defaults = default_params()
        for key, value in defaults.items():
            self.param_values[key] = value
            if key in self.param_vars:
                self.param_vars[key].set(value)
                self.param_labels[key].config(text=self._format_param_value(key, value))
        self.resize_method_var.set(DEFAULT_RESIZE_METHOD)
        self.refresh_view()

    def flip_horizontal(self) -> None:
        if self.base_image is None:
            return
        self.base_image = ImageOps.mirror(self.base_image)
        if self.source_array is not None:
            self.source_array = np.flip(self.source_array, axis=1)
        self.refresh_view()

    def flip_vertical(self) -> None:
        if self.base_image is None:
            return
        self.base_image = ImageOps.flip(self.base_image)
        if self.source_array is not None:
            self.source_array = np.flip(self.source_array, axis=0)
        self.refresh_view()

    def rotate_by(self, angle: int) -> None:
        current = self.param_values.get("rotate", 0)
        new_angle = max(-180, min(180, current + angle))
        self.param_values["rotate"] = new_angle
        if "rotate" in self.param_vars:
            self.param_vars["rotate"].set(new_angle)
            self.param_labels["rotate"].config(text=str(new_angle))
        self.refresh_view()

    def fit_view(self) -> None:
        self.main_canvas.fit_to_window()

    def actual_size_view(self) -> None:
        self.main_canvas.actual_size()

    def adjust_zoom(self, factor: float) -> None:
        self.main_canvas.set_zoom(self.main_canvas.get_zoom() * factor)

    def _on_magnifier_radius_change(self, val: str) -> None:
        value = int(float(val))
        self.magnifier_size.set(value)
        if self.magnifier_radius_label is not None:
            self.magnifier_radius_label.config(text=str(value))

    def _on_magnifier_zoom_change(self, val: str) -> None:
        value = int(float(val))
        self.magnifier_zoom.set(value)
        if self.magnifier_zoom_label is not None:
            self.magnifier_zoom_label.config(text=f"{value}x")

    def _format_export_dir(self) -> str:
        return str(self.export_dir)

    def _export_stem(self) -> str:
        if self.current_path is not None:
            return f"{self.current_path.stem}_processed"
        return "processed"

    @staticmethod
    def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
        candidate = directory / f"{stem}{suffix}"
        if not candidate.exists():
            return candidate
        index = 1
        while True:
            candidate = directory / f"{stem}_{index}{suffix}"
            if not candidate.exists():
                return candidate
            index += 1

    def choose_export_dir(self) -> None:
        path = filedialog.askdirectory(title="エクスポート先フォルダを選択", initialdir=str(self.export_dir))
        if not path:
            return
        self.export_dir = Path(path)
        if self.export_dir_label is not None:
            self.export_dir_label.config(text=self._format_export_dir())
        self.status_var.set(f"保存先: {self.export_dir}")

    def export_processed(self, fmt: str) -> None:
        image = self._get_processed_image()
        if image is None:
            messagebox.showinfo("エクスポート", "エクスポートする画像がありません。")
            return

        suffix_map = {"png": ".png", "jpeg": ".jpg", "npy": ".npy"}
        suffix = suffix_map.get(fmt)
        if suffix is None:
            return

        try:
            self.export_dir.mkdir(parents=True, exist_ok=True)
            out_path = self._unique_path(self.export_dir, self._export_stem(), suffix)
            self._write_export(image, out_path, fmt)
            self.status_var.set(f"エクスポートしました: {out_path}")
        except Exception as exc:
            messagebox.showerror("エクスポートエラー", str(exc))

    def export_processed_as(self) -> None:
        image = self._get_processed_image()
        if image is None:
            messagebox.showinfo("エクスポート", "エクスポートする画像がありません。")
            return

        path = filedialog.asksaveasfilename(
            title="処理後画像を保存",
            initialdir=str(self.export_dir),
            initialfile=f"{self._export_stem()}.png",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("NumPy", "*.npy"),
                ("すべて", "*.*"),
            ],
        )
        if not path:
            return

        out_path = Path(path)
        fmt = out_path.suffix.lower().lstrip(".")
        if fmt in ("jpg", "jpeg"):
            fmt = "jpeg"
        elif fmt != "npy":
            fmt = "png"

        try:
            self._write_export(image, out_path, fmt)
            self.export_dir = out_path.parent
            if self.export_dir_label is not None:
                self.export_dir_label.config(text=self._format_export_dir())
            self.status_var.set(f"エクスポートしました: {out_path}")
        except Exception as exc:
            messagebox.showerror("エクスポートエラー", str(exc))

    @staticmethod
    def _write_export(image: Image.Image, path: Path, fmt: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "npy":
            np.save(path, np.asarray(image))
            return
        if fmt == "jpeg":
            image.convert("RGB").save(path, format="JPEG", quality=95)
            return
        image.save(path, format="PNG")

    def _update_meta(self, path: Path, array: np.ndarray) -> None:
        stats = array_stats(array)
        lines = [
            f"path: {path.name}",
            f"type: {self.current_file_type}",
            f"shape: {stats['shape']}",
            f"dtype: {stats['dtype']}",
            f"min: {stats['min']:.4g}",
            f"max: {stats['max']:.4g}",
            f"mean: {stats['mean']:.4g}",
            f"std: {stats['std']:.4g}",
            f"size: {path.stat().st_size:,} bytes",
            f"no: {self.format_list_number(self.current_index)}",
            f"index: {self.current_index + 1}/{len(self.file_list)}",
        ]
        self.meta_text.config(state=tk.NORMAL)
        self.meta_text.delete("1.0", tk.END)
        self.meta_text.insert(tk.END, "\n".join(lines))
        self.meta_text.config(state=tk.DISABLED)

    def _on_main_cursor_move(self, x: int, y: int, event: tk.Event | None) -> None:
        if x < 0 or y < 0:
            self.magnifier.hide()
            self.status_var.set("準備完了")
            return

        pixel = self.main_canvas.get_pixel_value(x, y)
        zoom = self.main_canvas.get_zoom()
        self.status_var.set(f"({x}, {y})  value={pixel}  zoom={zoom:.2f}x")

        if not self.magnifier_enabled.get() or event is None:
            self.magnifier.hide()
            return

        patch = self.main_canvas.extract_patch(
            x, y, self.magnifier_size.get(), self.magnifier_zoom.get()
        )
        if patch is None:
            return
        sx = event.x_root
        sy = event.y_root
        self.magnifier.show_patch(patch, sx, sy)

    def show_shortcuts_help(self) -> None:
        text = (
            "ショートカット:\n"
            "  Ctrl+O : ファイルを開く\n"
            "  Ctrl+Shift+O : フォルダを開く\n"
            "  ←/→ : 前/次\n"
            "  R : ランダム\n"
            "  Home/End : 先頭/末尾\n"
            "  F : ウィンドウに合わせる\n"
            "  +/- : 拡大/縮小\n"
            "  中クリック/右ドラッグ : パン\n"
            "  ホイール : ズーム\n"
        )
        messagebox.showinfo("ショートカット", text)


def parse_cli_paths(argv: list[str]) -> list[Path]:
    roots: list[Path] = []
    for arg in argv:
        p = Path(arg)
        if not p.exists():
            continue
        if p.is_file() and is_supported_image(p):
            roots.append(p)
        elif p.is_dir() and directory_has_images(p):
            roots.append(p)
    return roots


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    style.configure("File.Treeview", font=("Consolas", 9))

    initial = parse_cli_paths(sys.argv[1:])
    app = ImageViewerApp(root, initial or None)
    root.mainloop()


if __name__ == "__main__":
    main()
