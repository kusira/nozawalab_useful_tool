"""NPY/RAW/PNG/JPEG画像の読み込みと画像処理パラメータをスライダーで調整するGUIアプリ。"""

from __future__ import annotations

import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.face_pipeline import (  # noqa: E402
    AFFINE_SIZE_HEIGHT,
    AFFINE_SIZE_WIDTH,
    bgr_to_pil,
    draw_dual_landmarks_overlay,
    first_face_landmarks,
    pil_to_bgr,
    warp_face_fs_affine,
)
from module.fa_landmark_calculator import resolve_torch_device  # noqa: E402

RAW_DTYPE_OPTIONS = [
    ("uint8", np.uint8),
    ("uint16", np.uint16),
    ("int16", np.int16),
    ("uint32", np.uint32),
    ("int32", np.int32),
    ("float32", np.float32),
    ("float64", np.float64),
]


class RawSettingsDialog(tk.Toplevel):
    """RAWファイル読み込み時のパラメータ入力ダイアログ。"""

    def __init__(
        self,
        parent: tk.Misc,
        path: Path,
        defaults: dict[str, object],
    ) -> None:
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


class ImageProcessingApp:
    PREVIEW_MAX_SIZE = 420

    PREVIEW_SPECS = (
        ("original", "元画像"),
        ("processed", "画像処理後"),
        ("affine_fa", "アフィン (face-alignment)"),
        ("affine_dlib", "アフィン (dlib)"),
    )

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NPY/RAW/PNG/JPEG画像プレビュー・特徴点解析")
        self.root.minsize(1280, 820)

        self.source_array: np.ndarray | None = None
        self.base_image: Image.Image | None = None
        self.cropped_image: Image.Image | None = None
        self.cropped_array: np.ndarray | None = None
        self.original_photo: ImageTk.PhotoImage | None = None
        self.processed_photo: ImageTk.PhotoImage | None = None
        self.preview_photos: dict[str, ImageTk.PhotoImage | None] = {key: None for key, _ in ImageProcessingApp.PREVIEW_SPECS}
        self.preview_labels: dict[str, ttk.Label] = {}
        self.preview_frames: dict[str, ttk.LabelFrame] = {}
        self.pipeline_images: dict[str, Image.Image | None] = {
            "processed": None,
            "processed_overlay": None,
            "affine_fa": None,
            "affine_dlib": None,
        }
        self._fa_calculator = None
        self._dlib_calculator = None
        self.landmark_device = resolve_torch_device()
        self.fa_enabled_var = tk.BooleanVar(value=True)
        self.dlib_enabled_var = tk.BooleanVar(value=True)
        self.last_landmark_timing_text: str | None = None
        self._landmark_thread: threading.Thread | None = None
        self._update_job: str | None = None
        self.current_path: Path | None = None
        self.current_file_type: str | None = None
        self.raw_settings: dict[str, object] = {
            "width": 1600,
            "height": 1300,
            "channels": 1,
            "offset": 0,
            "dtype": "uint16",
            "endian": "little",
        }

        self.params: dict[str, tk.IntVar] = {}
        self.value_labels: dict[str, ttk.Label] = {}
        self.crop_mode = False
        self.pending_crop: tuple[int, int, int, int] | None = None
        self._crop_drag_start: tuple[int, int] | None = None
        self.original_canvas: tk.Canvas | None = None
        self._original_canvas_photo: ImageTk.PhotoImage | None = None
        self._original_display_image: Image.Image | None = None
        self._original_canvas_scale_x = 1.0
        self._original_canvas_scale_y = 1.0
        self._original_canvas_offset_x = 0
        self._original_canvas_offset_y = 0
        self._original_canvas_display_size = (0, 0)
        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Button(top, text="ファイルを開く", command=self.open_file).pack(side=tk.LEFT)
        self.path_label = ttk.Label(top, text="ファイル未選択", width=70)
        self.path_label.pack(side=tk.LEFT, padx=(12, 0))
        self.reload_button = ttk.Button(top, text="RAW再読み込み", command=self.reload_raw, state=tk.DISABLED)
        self.reload_button.pack(side=tk.RIGHT, padx=(8, 0))
        self.save_preview_button = ttk.Button(
            top,
            text="プレビュー保存",
            command=self.save_pipeline_previews,
            state=tk.DISABLED,
        )
        self.save_preview_button.pack(side=tk.RIGHT, padx=(8, 0))
        landmark_opts = ttk.Frame(top)
        landmark_opts.pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Checkbutton(
            landmark_opts,
            text="FA",
            variable=self.fa_enabled_var,
            command=self._on_landmark_engine_toggle,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            landmark_opts,
            text="dlib",
            variable=self.dlib_enabled_var,
            command=self._on_landmark_engine_toggle,
        ).pack(side=tk.LEFT, padx=(6, 0))
        self.landmark_button = ttk.Button(
            top,
            text="特徴点算出",
            command=self.run_landmark_analysis,
            state=tk.DISABLED,
        )
        self.landmark_button.pack(side=tk.RIGHT, padx=(8, 0))
        self.landmark_timing_label = ttk.Label(top, text="")
        self.landmark_timing_label.pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(top, text="全設定リセット", command=self.reset_all).pack(side=tk.RIGHT)

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        controls = ttk.Frame(main, padding=8, width=320)
        main.add(controls, weight=0)

        preview = ttk.Frame(main, padding=8)
        main.add(preview, weight=1)

        self._build_controls(controls)
        self._build_preview(preview)

    def _build_controls(self, parent: ttk.Frame) -> None:
        preprocess = ttk.LabelFrame(parent, text="前処理", padding=6)
        preprocess.pack(fill=tk.X, pady=(0, 8))

        flip_buttons = ttk.Frame(preprocess)
        flip_buttons.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(flip_buttons, text="左右反転", command=self.flip_original_horizontal).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3)
        )
        ttk.Button(flip_buttons, text="上下反転", command=self.flip_original_vertical).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0)
        )

        ttk.Label(
            preprocess,
            text="範囲選択は任意です。未適用の場合は元画像全体で画像処理・特徴点算出を行います。",
            wraplength=280,
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(4, 4))

        crop_buttons = ttk.Frame(preprocess)
        crop_buttons.pack(fill=tk.X, pady=(0, 3))
        self.crop_select_button = ttk.Button(
            crop_buttons,
            text="範囲選択",
            command=self.start_crop_mode,
            state=tk.DISABLED,
        )
        self.crop_select_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        self.crop_apply_button = ttk.Button(
            crop_buttons,
            text="適用",
            command=self.apply_crop,
            state=tk.DISABLED,
        )
        self.crop_apply_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        crop_buttons2 = ttk.Frame(preprocess)
        crop_buttons2.pack(fill=tk.X, pady=(0, 4))
        self.crop_cancel_button = ttk.Button(
            crop_buttons2,
            text="取消",
            command=self.cancel_crop_mode,
            state=tk.DISABLED,
        )
        self.crop_cancel_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        self.crop_reset_button = ttk.Button(
            crop_buttons2,
            text="リセット",
            command=self.reset_crop,
            state=tk.DISABLED,
        )
        self.crop_reset_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        ttk.Label(
            preprocess,
            text="トリミング後の縮尺 (0.1〜1.0)",
            wraplength=280,
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(2, 2))
        self._add_param_scale_row(preprocess, "resize", "リサイズ", 1, 10, 10)
        ttk.Label(
            preprocess,
            text="YOLO 検出 bbox の拡大倍率 (1.0 = 原寸)",
            wraplength=280,
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(2, 2))
        self._add_param_scale_row(preprocess, "yolo_bbox", "YOLO bbox", 10, 30, 10)

        ttk.Label(parent, text="画像処理", font=("", 11, "bold")).pack(anchor=tk.W, pady=(0, 4))

        slider_defs = [
            ("brightness", "明るさ", 0, 200, 100),
            ("contrast", "コントラスト", 0, 200, 100),
            ("gamma", "ガンマ", 10, 300, 100),
            ("blur", "ぼかし", 0, 20, 0),
            ("sharpen", "シャープ", 0, 100, 0),
            ("threshold", "二値化しきい値", 0, 255, 0),
            ("rotate", "回転 (度)", 0, 360, 0),
            ("clip_min", "表示下限 (%)", 0, 100, 0),
            ("clip_max", "表示上限 (%)", 0, 100, 100),
            ("equalize", "ヒストグラム均等化", 0, 100, 0),
            ("invert", "反転", 0, 100, 0),
        ]

        for key, label, from_, to, default in slider_defs:
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=2)

            ttk.Label(frame, text=label, width=18).pack(side=tk.LEFT)
            var = tk.IntVar(value=default)
            self.params[key] = var
            value_label = ttk.Label(frame, text=self._format_param_value(key, default), width=5)
            value_label.pack(side=tk.RIGHT)
            self.value_labels[key] = value_label

            scale = ttk.Scale(
                frame,
                from_=from_,
                to=to,
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda _value, k=key: self._on_slider_change(k),
            )
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))

        meta_frame = ttk.LabelFrame(parent, text="配列情報", padding=6)
        meta_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.meta_text = tk.Text(meta_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.meta_text.pack(fill=tk.BOTH, expand=True)

    def _add_param_scale_row(
        self,
        parent: ttk.Frame,
        key: str,
        label: str,
        from_: int,
        to: int,
        default: int,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label, width=12).pack(side=tk.LEFT)
        var = tk.IntVar(value=default)
        self.params[key] = var
        value_label = ttk.Label(frame, text=self._format_param_value(key, default), width=5)
        value_label.pack(side=tk.RIGHT)
        self.value_labels[key] = value_label
        ttk.Scale(
            frame,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=var,
            command=lambda _value, k=key: self._on_slider_change(k),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))

    def _build_preview(self, parent: ttk.Frame) -> None:
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.BOTH, expand=True)
        for col in range(2):
            grid.columnconfigure(col, weight=1)
        for row in range(2):
            grid.rowconfigure(row, weight=1)

        positions = {
            "original": (0, 0),
            "processed": (0, 1),
            "affine_fa": (1, 0),
            "affine_dlib": (1, 1),
        }
        for key, title in self.PREVIEW_SPECS:
            row, col = positions[key]
            frame = ttk.LabelFrame(grid, text=title, padding=8)
            frame.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
            self.preview_frames[key] = frame
            if key == "original":
                canvas = tk.Canvas(frame, highlightthickness=0, bg="#2b2b2b", cursor="arrow")
                canvas.pack(fill=tk.BOTH, expand=True)
                canvas.bind("<Configure>", self._on_original_canvas_resize)
                self.original_canvas = canvas
                self.preview_labels[key] = canvas
            else:
                label = ttk.Label(frame, anchor=tk.CENTER)
                label.pack(fill=tk.BOTH, expand=True)
                self.preview_labels[key] = label
            if key in ("affine_fa", "affine_dlib"):
                frame.config(text=f"{title} ({AFFINE_SIZE_WIDTH} × {AFFINE_SIZE_HEIGHT})")

        self.original_label = self.preview_labels["original"]
        self.processed_label = self.preview_labels["processed"]

    def _on_slider_change(self, key: str) -> None:
        value = self.params[key].get()
        if key in ("resize", "yolo_bbox"):
            minimum = 1 if key == "resize" else 10
            maximum = 10 if key == "resize" else 30
            value = max(minimum, min(maximum, round(value)))
        self._set_param(key, int(value), update_preview=key != "yolo_bbox")

    def _format_param_value(self, key: str, value: int) -> str:
        if key in ("resize", "yolo_bbox"):
            return f"{value / 10:.1f}"
        return str(value)

    def _set_param(self, key: str, value: int, *, update_preview: bool = False) -> None:
        self.params[key].set(value)
        self.value_labels[key].config(text=self._format_param_value(key, value))
        if update_preview:
            self.schedule_update()

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="画像ファイルを選択",
            filetypes=[
                ("対応形式", "*.npy *.raw *.png *.jpg *.jpeg"),
                ("NumPy配列", "*.npy"),
                ("RAW画像", "*.raw"),
                ("PNG画像", "*.png"),
                ("JPEG画像", "*.jpg *.jpeg"),
                ("すべてのファイル", "*.*"),
            ],
        )
        if not path:
            return

        file_path = Path(path)
        suffix = file_path.suffix.lower()

        try:
            if suffix == ".npy":
                array = self.load_npy(file_path)
                image = self.array_to_image(array)
                self.current_file_type = "npy"
                self.reload_button.config(state=tk.DISABLED)
            elif suffix == ".png":
                array, image = self.load_raster(file_path)
                self.current_file_type = "png"
                self.reload_button.config(state=tk.DISABLED)
            elif suffix in (".jpg", ".jpeg"):
                array, image = self.load_raster(file_path)
                self.current_file_type = "jpeg"
                self.reload_button.config(state=tk.DISABLED)
            else:
                settings = self._ask_raw_settings(file_path)
                if settings is None:
                    return
                self.raw_settings = settings
                array = self.load_raw(file_path, settings)
                image = self.array_to_image(array)
                self.current_file_type = "raw"
                self.reload_button.config(state=tk.NORMAL)
        except Exception as exc:
            messagebox.showerror("読み込みエラー", f"ファイルを読み込めませんでした。\n{exc}")
            return

        self.current_path = file_path
        self.source_array = array
        self.base_image = image
        self.cropped_image = None
        self.cropped_array = None
        self.cancel_crop_mode()
        self.path_label.config(text=str(file_path))
        self._update_meta(array, file_path)
        self.reset_params(update_preview=False)
        self._clear_affine_preview()
        self.landmark_button.config(state=tk.NORMAL)
        self.save_preview_button.config(state=tk.DISABLED)
        self.crop_select_button.config(state=tk.NORMAL)
        self._update_crop_reset_button_state()
        self.update_preview()

    def reload_raw(self) -> None:
        if self.current_path is None or self.current_file_type != "raw":
            return

        settings = self._ask_raw_settings(self.current_path)
        if settings is None:
            return

        try:
            self.raw_settings = settings
            array = self.load_raw(self.current_path, settings)
            image = self.array_to_image(array)
        except Exception as exc:
            messagebox.showerror("読み込みエラー", f"RAWファイルを読み込めませんでした。\n{exc}")
            return

        self.source_array = array
        self.base_image = image
        self.cropped_image = None
        self.cropped_array = None
        self.cancel_crop_mode()
        self._update_crop_reset_button_state()
        self._update_meta(array, self.current_path)
        self._clear_affine_preview()
        self.save_preview_button.config(state=tk.DISABLED)
        self.update_preview()

    def _ask_raw_settings(self, path: Path) -> dict[str, object] | None:
        guessed = self._guess_raw_settings(path, self.raw_settings)
        dialog = RawSettingsDialog(self.root, path, guessed)
        self.root.wait_window(dialog)
        return dialog.result

    @staticmethod
    def load_npy(path: Path) -> np.ndarray:
        return np.load(path)

    @staticmethod
    def load_raster(path: Path) -> tuple[np.ndarray, Image.Image]:
        with Image.open(path) as img:
            image = img.convert("RGB" if img.mode not in ("L", "LA") else "L")
        return np.asarray(image), image

    load_png = load_raster

    @staticmethod
    def _resolve_raw_dtype(dtype_name: str, endian: str) -> np.dtype:
        base_dtype = next(dt for name, dt in RAW_DTYPE_OPTIONS if name == dtype_name)
        np_dtype = np.dtype(base_dtype)
        if np_dtype.itemsize > 1:
            byteorder = "<" if endian == "little" else ">"
            np_dtype = np_dtype.newbyteorder(byteorder)
        return np_dtype

    @staticmethod
    def load_raw(path: Path, settings: dict[str, object]) -> np.ndarray:
        width = int(settings["width"])
        height = int(settings["height"])
        channels = int(settings["channels"])
        offset = int(settings["offset"])
        dtype_name = str(settings["dtype"])
        endian = str(settings["endian"])

        dtype = ImageProcessingApp._resolve_raw_dtype(dtype_name, endian)

        file_size = path.stat().st_size
        if offset >= file_size:
            raise ValueError("オフセットがファイルサイズ以上です。")

        count = width * height * channels
        available_bytes = file_size - offset
        required_bytes = count * dtype.itemsize
        if available_bytes < required_bytes:
            raise ValueError(
                f"データサイズが不足しています。必要: {required_bytes:,} bytes, 利用可能: {available_bytes:,} bytes"
            )

        data = np.fromfile(path, dtype=dtype, count=count, offset=offset)
        if channels == 1:
            return data.reshape(height, width)
        return data.reshape(height, width, channels)

    @staticmethod
    def _guess_raw_settings(path: Path, current: dict[str, object]) -> dict[str, object]:
        file_size = path.stat().st_size
        offset = int(current.get("offset", 0))
        dtype_name = str(current.get("dtype", "uint16"))
        dtype = np.dtype(next(dt for name, dt in RAW_DTYPE_OPTIONS if name == dtype_name))
        channels = max(1, int(current.get("channels", 1)))
        available = max(file_size - offset, 0)
        element_count = available // dtype.itemsize if available else 0

        width = int(current.get("width", 1600))
        height = int(current.get("height", 1300))
        if width > 0 and height > 0 and width * height * channels == element_count:
            return {
                "width": width,
                "height": height,
                "channels": channels,
                "offset": offset,
                "dtype": dtype_name,
                "endian": current.get("endian", "little"),
            }

        common_sizes = [256, 512, 640, 720, 768, 1024, 1280, 1920, 2048]
        for side in common_sizes:
            if side * side * channels * dtype.itemsize + offset == file_size:
                return {
                    "width": side,
                    "height": side,
                    "channels": channels,
                    "offset": offset,
                    "dtype": dtype_name,
                    "endian": current.get("endian", "little"),
                }

        if element_count > 0:
            side = int(np.sqrt(element_count // channels))
            if side > 0 and side * side * channels == element_count:
                width = side
                height = side

        return {
            "width": width,
            "height": height,
            "channels": channels,
            "offset": offset,
            "dtype": dtype_name,
            "endian": current.get("endian", "little"),
        }

    def reset_params(self, update_preview: bool = True) -> None:
        defaults = {
            "resize": 10,
            "yolo_bbox": 10,
            "brightness": 100,
            "contrast": 100,
            "gamma": 100,
            "blur": 0,
            "sharpen": 0,
            "threshold": 0,
            "rotate": 0,
            "clip_min": 0,
            "clip_max": 100,
            "equalize": 0,
            "invert": 0,
        }
        for key, value in defaults.items():
            self._set_param(key, value)

        if update_preview:
            self.update_preview()

    def reset_all(self) -> None:
        """スライダー・トリミング・特徴点結果をまとめて初期化する。"""
        self.cancel_crop_mode()
        self.cropped_image = None
        self.cropped_array = None
        self.pending_crop = None
        self._update_crop_reset_button_state()
        self.reset_params(update_preview=False)
        self._clear_affine_preview()
        self.save_preview_button.config(state=tk.DISABLED)
        self.last_landmark_timing_text = None
        self._update_landmark_timing_label()
        if self.base_image is not None:
            self.update_preview()

    def flip_original_horizontal(self) -> None:
        if self.base_image is None:
            return
        self.pending_crop = None
        self.cropped_image = None
        self.cropped_array = None
        self.base_image = ImageOps.mirror(self.base_image)
        if self.source_array is not None:
            self.source_array = np.flip(self.source_array, axis=1)
        if self.crop_mode:
            self.crop_apply_button.config(state=tk.DISABLED)
            self._render_original_canvas(self.base_image)
            self._set_preview_frame_title("original", self.base_image)
        else:
            self._update_crop_reset_button_state()
            self.update_preview()

    def flip_original_vertical(self) -> None:
        if self.base_image is None:
            return
        self.pending_crop = None
        self.cropped_image = None
        self.cropped_array = None
        self.base_image = ImageOps.flip(self.base_image)
        if self.source_array is not None:
            self.source_array = np.flip(self.source_array, axis=0)
        if self.crop_mode:
            self.crop_apply_button.config(state=tk.DISABLED)
            self._render_original_canvas(self.base_image)
            self._set_preview_frame_title("original", self.base_image)
        else:
            self._update_crop_reset_button_state()
            self.update_preview()

    def start_crop_mode(self) -> None:
        if self.base_image is None:
            messagebox.showwarning("トリミング", "先に画像を開いてください。")
            return
        self.crop_mode = True
        self.pending_crop = None
        self._crop_drag_start = None
        self.crop_select_button.config(state=tk.DISABLED)
        self.crop_apply_button.config(state=tk.DISABLED)
        self.crop_cancel_button.config(state=tk.NORMAL)
        if self.original_canvas is not None:
            self.original_canvas.config(cursor="crosshair")
        self._bind_crop_events()
        self._render_original_canvas(self.base_image)
        self._set_preview_frame_title("original", self.base_image)

    def cancel_crop_mode(self) -> None:
        was_crop_mode = self.crop_mode
        self.crop_mode = False
        self.pending_crop = None
        self._crop_drag_start = None
        self._unbind_crop_events()
        if self.original_canvas is not None:
            self.original_canvas.config(cursor="arrow")
        self.crop_apply_button.config(state=tk.DISABLED)
        self.crop_cancel_button.config(state=tk.DISABLED)
        if self.base_image is not None:
            self.crop_select_button.config(state=tk.NORMAL)
        else:
            self.crop_select_button.config(state=tk.DISABLED)
        if was_crop_mode and self.base_image is not None:
            self.update_preview()

    def _update_crop_reset_button_state(self) -> None:
        if self.cropped_image is not None:
            self.crop_reset_button.config(state=tk.NORMAL)
        else:
            self.crop_reset_button.config(state=tk.DISABLED)

    def reset_crop(self) -> None:
        if self.base_image is None:
            return
        if self.cropped_image is None and not self.crop_mode and self.pending_crop is None:
            return

        self.cancel_crop_mode()
        self.cropped_image = None
        self.cropped_array = None
        self.pending_crop = None
        self._clear_affine_preview()
        self.save_preview_button.config(state=tk.DISABLED)
        self._update_crop_reset_button_state()
        self.update_preview()

    def apply_crop(self) -> None:
        if self.base_image is None or self.pending_crop is None:
            return

        left, top, right, bottom = self.pending_crop
        if right - left < 2 or bottom - top < 2:
            messagebox.showwarning("トリミング", "トリミング範囲が小さすぎます。")
            return

        self.cropped_image = self.base_image.crop((left, top, right, bottom))
        if self.source_array is not None:
            self.cropped_array = self._crop_source_array(self.source_array, left, top, right, bottom)
        else:
            self.cropped_array = None

        self.pending_crop = None
        self._clear_affine_preview()
        self.save_preview_button.config(state=tk.DISABLED)
        self.cancel_crop_mode()
        self._update_crop_reset_button_state()

    @staticmethod
    def _crop_source_array(array: np.ndarray, left: int, top: int, right: int, bottom: int) -> np.ndarray:
        cropped = array[top:bottom, left:right]
        return np.ascontiguousarray(cropped)

    def _bind_crop_events(self) -> None:
        if self.original_canvas is None:
            return
        self.original_canvas.bind("<ButtonPress-1>", self._on_crop_press)
        self.original_canvas.bind("<B1-Motion>", self._on_crop_drag)
        self.original_canvas.bind("<ButtonRelease-1>", self._on_crop_release)

    def _unbind_crop_events(self) -> None:
        if self.original_canvas is None:
            return
        self.original_canvas.unbind("<ButtonPress-1>")
        self.original_canvas.unbind("<B1-Motion>")
        self.original_canvas.unbind("<ButtonRelease-1>")

    def _on_original_canvas_resize(self, _event: tk.Event) -> None:
        if self.base_image is None:
            return
        self._render_original_canvas(self.base_image)
        self._set_preview_frame_title("original", self.base_image)

    def _on_crop_press(self, event: tk.Event) -> None:
        if not self.crop_mode or self.base_image is None:
            return
        self._crop_drag_start = (event.x, event.y)
        if self.original_canvas is not None:
            self.original_canvas.delete("crop_drag")
            self.original_canvas.delete("crop_rect")

    def _on_crop_drag(self, event: tk.Event) -> None:
        if not self.crop_mode or self._crop_drag_start is None or self.original_canvas is None:
            return
        x0, y0 = self._crop_drag_start
        self.original_canvas.delete("crop_drag")
        self.original_canvas.create_rectangle(
            x0,
            y0,
            event.x,
            event.y,
            outline="#00ff88",
            width=2,
            dash=(4, 4),
            tags="crop_drag",
        )

    def _on_crop_release(self, event: tk.Event) -> None:
        if not self.crop_mode or self._crop_drag_start is None or self.base_image is None:
            return

        if self.original_canvas is not None:
            self.original_canvas.delete("crop_drag")

        start = self._canvas_to_image_point(self._crop_drag_start[0], self._crop_drag_start[1])
        end = self._canvas_to_image_point(event.x, event.y)
        self._crop_drag_start = None
        if start is None or end is None:
            return

        left = max(0, min(start[0], end[0]))
        top = max(0, min(start[1], end[1]))
        right = min(self.base_image.width, max(start[0], end[0]))
        bottom = min(self.base_image.height, max(start[1], end[1]))
        if right - left < 2 or bottom - top < 2:
            self.pending_crop = None
            self.crop_apply_button.config(state=tk.DISABLED)
            return

        self.pending_crop = (left, top, right, bottom)
        self.crop_apply_button.config(state=tk.NORMAL)
        self._draw_crop_overlay()

    def _canvas_to_image_point(self, canvas_x: int, canvas_y: int) -> tuple[int, int] | None:
        if self._original_display_image is None:
            return None

        disp_w, disp_h = self._original_canvas_display_size
        if disp_w <= 0 or disp_h <= 0:
            return None

        local_x = canvas_x - self._original_canvas_offset_x
        local_y = canvas_y - self._original_canvas_offset_y
        local_x = max(0, min(local_x, disp_w))
        local_y = max(0, min(local_y, disp_h))

        image_x = int(round(local_x * self._original_canvas_scale_x))
        image_y = int(round(local_y * self._original_canvas_scale_y))
        image_x = max(0, min(image_x, self._original_display_image.width))
        image_y = max(0, min(image_y, self._original_display_image.height))
        return image_x, image_y

    def _image_to_canvas_point(self, image_x: int, image_y: int) -> tuple[int, int]:
        canvas_x = self._original_canvas_offset_x + image_x / self._original_canvas_scale_x
        canvas_y = self._original_canvas_offset_y + image_y / self._original_canvas_scale_y
        return int(round(canvas_x)), int(round(canvas_y))

    def _draw_crop_overlay(self) -> None:
        if self.original_canvas is None or self.pending_crop is None:
            return

        left, top, right, bottom = self.pending_crop
        x0, y0 = self._image_to_canvas_point(left, top)
        x1, y1 = self._image_to_canvas_point(right, bottom)
        self.original_canvas.delete("crop_rect")
        self.original_canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            outline="#00ff88",
            width=2,
            tags="crop_rect",
        )

    def _render_original_canvas(self, image: Image.Image) -> None:
        if self.original_canvas is None:
            return

        self._original_display_image = image
        self.original_canvas.update_idletasks()
        canvas_w = max(self.original_canvas.winfo_width(), 1)
        canvas_h = max(self.original_canvas.winfo_height(), 1)
        fit_w = min(canvas_w, self.PREVIEW_MAX_SIZE)
        fit_h = min(canvas_h, self.PREVIEW_MAX_SIZE)

        preview = image.copy()
        preview.thumbnail((fit_w, fit_h), Image.Resampling.LANCZOS)
        disp_w, disp_h = preview.size
        if disp_w <= 0 or disp_h <= 0:
            return

        self._original_canvas_scale_x = image.width / disp_w
        self._original_canvas_scale_y = image.height / disp_h
        self._original_canvas_display_size = (disp_w, disp_h)
        self._original_canvas_offset_x = (canvas_w - disp_w) // 2
        self._original_canvas_offset_y = (canvas_h - disp_h) // 2

        self._original_canvas_photo = ImageTk.PhotoImage(preview)
        self.original_canvas.delete("all")
        self.original_canvas.create_image(
            self._original_canvas_offset_x,
            self._original_canvas_offset_y,
            anchor=tk.NW,
            image=self._original_canvas_photo,
            tags="preview_image",
        )
        self._draw_crop_overlay()

    def schedule_update(self) -> None:
        if self._update_job is not None:
            self.root.after_cancel(self._update_job)
        self._update_job = self.root.after(80, self._run_scheduled_update)

    def _run_scheduled_update(self) -> None:
        self._update_job = None
        self.update_preview()

    def update_preview(self) -> None:
        if self.base_image is None:
            return

        self._render_original_canvas(self.base_image)
        self._set_preview_frame_title("original", self.base_image)

        processed = self._get_processed_image()
        if processed is None:
            self._set_processed_placeholder("画像を開いてください")
            return

        self.pipeline_images["processed"] = processed
        self.pipeline_images["processed_overlay"] = None
        self._set_preview_image("processed", processed)
        if not self.crop_mode:
            self._clear_affine_preview()

    def _set_processed_placeholder(self, text: str) -> None:
        self.preview_photos["processed"] = None
        self.preview_labels["processed"].config(image="", text=text)
        self._set_preview_frame_title("processed", None)

    def _get_resize_scale(self) -> float:
        return self.params["resize"].get() / 10.0

    def _scale_image(self, image: Image.Image) -> Image.Image:
        scale = self._get_resize_scale()
        if scale >= 0.999:
            return image.copy()
        new_w = max(1, int(round(image.width * scale)))
        new_h = max(1, int(round(image.height * scale)))
        if new_w == image.width and new_h == image.height:
            return image.copy()
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _get_working_image(self) -> Image.Image | None:
        """トリミング適用済みならその画像、未適用なら元画像を返す。"""
        if self.cropped_image is not None:
            return self.cropped_image
        return self.base_image

    def _get_resized_crop(self) -> Image.Image | None:
        working = self._get_working_image()
        if working is None:
            return None
        return self._scale_image(working.copy())

    def _get_processed_image(self) -> Image.Image | None:
        resized = self._get_resized_crop()
        if resized is None:
            return None
        return self.apply_processing(resized)

    @staticmethod
    def _format_image_resolution(image: Image.Image) -> str:
        return f"{image.width} × {image.height}"

    def _preview_base_title(self, key: str) -> str:
        for preview_key, title in self.PREVIEW_SPECS:
            if preview_key == key:
                return title
        return key

    def _affine_preview_title(self, key: str, image: Image.Image | None = None) -> str:
        base = self._preview_base_title(key)
        if image is not None:
            return f"{base} ({self._format_image_resolution(image)})"
        return f"{base} ({AFFINE_SIZE_WIDTH} × {AFFINE_SIZE_HEIGHT})"

    def _set_preview_frame_title(self, key: str, image: Image.Image | None) -> None:
        frame = self.preview_frames.get(key)
        if frame is None:
            return
        if key in ("original", "processed") and image is not None:
            frame.config(text=f"{self._preview_base_title(key)} ({self._format_image_resolution(image)})")
        elif key in ("affine_fa", "affine_dlib"):
            frame.config(text=self._affine_preview_title(key, image))
        else:
            frame.config(text=self._preview_base_title(key))

    def _set_preview_image(self, key: str, image: Image.Image) -> None:
        if key == "original":
            self._render_original_canvas(image)
            self._set_preview_frame_title(key, image)
            return

        photo = self.to_photo(image)
        self.preview_photos[key] = photo
        label = self.preview_labels[key]
        label.config(image=photo, text="")
        if key in ("original", "processed", "affine_fa", "affine_dlib"):
            self._set_preview_frame_title(key, image)

    def _affine_preview_placeholder(self, engine: str) -> str:
        if engine == "fa":
            return "FA: 無効" if not self.fa_enabled_var.get() else "「特徴点算出」で生成"
        return "dlib: 無効" if not self.dlib_enabled_var.get() else "「特徴点算出」で生成"

    def _on_landmark_engine_toggle(self) -> None:
        self._clear_affine_preview()
        if self.pipeline_images.get("processed_overlay") is not None:
            processed = self.pipeline_images.get("processed")
            if processed is not None:
                self._set_preview_image("processed", processed)
            self.pipeline_images["processed_overlay"] = None
            self.save_preview_button.config(state=tk.DISABLED)

    def _clear_affine_preview(self) -> None:
        for key, engine in (("affine_fa", "fa"), ("affine_dlib", "dlib")):
            self.pipeline_images[key] = None
            self.preview_photos[key] = None
            self.preview_labels[key].config(image="", text=self._affine_preview_placeholder(engine))
            self._set_preview_frame_title(key, None)

    def _preload_landmark_models(self, *, fa_enabled: bool, dlib_enabled: bool) -> None:
        """推論計測にモデル読み込み時間を含めないよう、算出前にモデルを読み込む。"""
        self.landmark_button.config(text="準備中...")
        self.root.update_idletasks()
        if fa_enabled:
            self._get_fa_calculator()
        if dlib_enabled:
            self._get_dlib_calculator()

    def _format_landmark_timing(
        self,
        *,
        fa_enabled: bool,
        dlib_enabled: bool,
        fa_elapsed: float | None,
        dlib_elapsed: float | None,
        yolo_elapsed: float | None,
        total_elapsed: float,
    ) -> str:
        parts: list[str] = []
        if fa_enabled and fa_elapsed is not None:
            parts.append(f"FA {fa_elapsed:.2f}s")
        if dlib_enabled and dlib_elapsed is not None:
            parts.append(f"dlib {dlib_elapsed:.2f}s")
        if yolo_elapsed is not None and yolo_elapsed > 0.0:
            parts.append(f"YOLO {yolo_elapsed:.2f}s")
        if not parts:
            return f"特徴点算出: 合計 {total_elapsed:.2f}s"
        return f"特徴点算出: {', '.join(parts)} (合計 {total_elapsed:.2f}s)"

    def _get_yolo_bbox_scale(self) -> float:
        return self.params["yolo_bbox"].get() / 10.0

    def _update_landmark_timing_label(self) -> None:
        if self.last_landmark_timing_text:
            self.landmark_timing_label.config(text=self.last_landmark_timing_text)
        else:
            self.landmark_timing_label.config(text="")

    def _get_fa_calculator(self):
        if self._fa_calculator is None:
            from module.fa_landmark_calculator import FaceLandmarkCalculator

            self._fa_calculator = FaceLandmarkCalculator(landmark_type="2D", device=self.landmark_device)
        return self._fa_calculator

    def _get_dlib_calculator(self):
        if self._dlib_calculator is None:
            from module.dlib_landmark_calclator import DlibLandmarkCalculator

            self._dlib_calculator = DlibLandmarkCalculator(device=self.landmark_device)
        return self._dlib_calculator

    def run_landmark_analysis(self) -> None:
        if self.base_image is None:
            messagebox.showwarning("特徴点算出", "先に画像を開いてください。")
            return
        if self._landmark_thread is not None and self._landmark_thread.is_alive():
            return

        fa_enabled = self.fa_enabled_var.get()
        dlib_enabled = self.dlib_enabled_var.get()
        if not fa_enabled and not dlib_enabled:
            messagebox.showwarning("特徴点算出", "FA または dlib のいずれかを有効にしてください。")
            return

        self.landmark_button.config(state=tk.DISABLED, text="準備中...")
        try:
            self._preload_landmark_models(fa_enabled=fa_enabled, dlib_enabled=dlib_enabled)
        except Exception as exc:
            self.landmark_button.config(state=tk.NORMAL, text="特徴点算出")
            messagebox.showerror("特徴点算出", f"モデルの読み込みに失敗しました。\n{exc}")
            return

        processed = self._get_processed_image()
        if processed is None:
            messagebox.showwarning("特徴点算出", "処理対象の画像がありません。")
            return
        processed_bgr = pil_to_bgr(processed)
        self.pipeline_images["processed"] = processed
        self._set_preview_image("processed", processed)
        self.landmark_button.config(text="算出中...")
        bbox_scale = self._get_yolo_bbox_scale()

        def worker() -> None:
            errors: list[str] = []
            processed_overlay: Image.Image | None = None
            affine_fa_image: Image.Image | None = None
            affine_dlib_image: Image.Image | None = None
            fa_result: dict | None = None
            dlib_result: dict | None = None
            fa_elapsed: float | None = None
            dlib_elapsed: float | None = None
            yolo_elapsed = 0.0
            started_at = time.perf_counter()

            if fa_enabled:
                try:
                    fa_result, fa_timing = self._get_fa_calculator().predict_from_bgr_timed(
                        processed_bgr, bbox_scale=bbox_scale
                    )
                    fa_elapsed = fa_timing["landmark"]
                    yolo_elapsed += fa_timing["yolo"]
                except Exception as exc:
                    errors.append(f"face-alignment: {exc}")

            if dlib_enabled:
                try:
                    dlib_result, dlib_timing = self._get_dlib_calculator().predict_from_bgr_timed(
                        processed_bgr, bbox_scale=bbox_scale
                    )
                    dlib_elapsed = dlib_timing["landmark"]
                    yolo_elapsed += dlib_timing["yolo"]
                except Exception as exc:
                    errors.append(f"dlib: {exc}")

            if fa_result is not None or dlib_result is not None:
                overlay_bgr = draw_dual_landmarks_overlay(processed_bgr, fa_result, dlib_result)
                processed_overlay = bgr_to_pil(overlay_bgr)

            if fa_enabled:
                fa_landmarks = first_face_landmarks(fa_result) if fa_result is not None else None
                if fa_landmarks is not None:
                    affine_bgr = warp_face_fs_affine(processed_bgr, fa_landmarks)
                    if affine_bgr is not None:
                        affine_fa_image = bgr_to_pil(affine_bgr)

            if dlib_enabled:
                dlib_landmarks = first_face_landmarks(dlib_result) if dlib_result is not None else None
                if dlib_landmarks is not None:
                    affine_bgr = warp_face_fs_affine(processed_bgr, dlib_landmarks)
                    if affine_bgr is not None:
                        affine_dlib_image = bgr_to_pil(affine_bgr)

            total_elapsed = time.perf_counter() - started_at

            self.root.after(
                0,
                lambda: self._on_landmark_analysis_done(
                    fa_enabled=fa_enabled,
                    dlib_enabled=dlib_enabled,
                    fa_elapsed=fa_elapsed,
                    dlib_elapsed=dlib_elapsed,
                    yolo_elapsed=yolo_elapsed,
                    total_elapsed=total_elapsed,
                    processed_overlay=processed_overlay,
                    affine_fa_image=affine_fa_image,
                    affine_dlib_image=affine_dlib_image,
                    errors=errors,
                ),
            )

        self._landmark_thread = threading.Thread(target=worker, daemon=True, name="landmark-analysis")
        self._landmark_thread.start()

    def _on_landmark_analysis_done(
        self,
        *,
        fa_enabled: bool,
        dlib_enabled: bool,
        fa_elapsed: float | None,
        dlib_elapsed: float | None,
        yolo_elapsed: float,
        total_elapsed: float,
        processed_overlay: Image.Image | None,
        affine_fa_image: Image.Image | None,
        affine_dlib_image: Image.Image | None,
        errors: list[str],
    ) -> None:
        self.landmark_button.config(state=tk.NORMAL, text="特徴点算出")
        self._landmark_thread = None
        self.last_landmark_timing_text = self._format_landmark_timing(
            fa_enabled=fa_enabled,
            dlib_enabled=dlib_enabled,
            fa_elapsed=fa_elapsed,
            dlib_elapsed=dlib_elapsed,
            yolo_elapsed=yolo_elapsed,
            total_elapsed=total_elapsed,
        )
        self._update_landmark_timing_label()

        if processed_overlay is not None:
            self.pipeline_images["processed_overlay"] = processed_overlay
            self._set_preview_image("processed", processed_overlay)
        elif fa_enabled or dlib_enabled:
            self.preview_labels["processed"].config(image="", text="特徴点検出失敗")
            processed_ref = self.pipeline_images.get("processed")
            if processed_ref is not None:
                self._set_preview_frame_title("processed", processed_ref)

        if not fa_enabled:
            self.pipeline_images["affine_fa"] = None
            self.preview_labels["affine_fa"].config(image="", text="FA: 無効")
            self._set_preview_frame_title("affine_fa", None)
        elif affine_fa_image is not None:
            self.pipeline_images["affine_fa"] = affine_fa_image
            self._set_preview_image("affine_fa", affine_fa_image)
        else:
            self.pipeline_images["affine_fa"] = None
            self.preview_labels["affine_fa"].config(image="", text="FA: アフィン変換不可")
            self._set_preview_frame_title("affine_fa", None)

        if not dlib_enabled:
            self.pipeline_images["affine_dlib"] = None
            self.preview_labels["affine_dlib"].config(image="", text="dlib: 無効")
            self._set_preview_frame_title("affine_dlib", None)
        elif affine_dlib_image is not None:
            self.pipeline_images["affine_dlib"] = affine_dlib_image
            self._set_preview_image("affine_dlib", affine_dlib_image)
        else:
            self.pipeline_images["affine_dlib"] = None
            self.preview_labels["affine_dlib"].config(image="", text="dlib: アフィン変換不可")
            self._set_preview_frame_title("affine_dlib", None)

        if processed_overlay is not None or affine_fa_image is not None or affine_dlib_image is not None:
            self.save_preview_button.config(state=tk.NORMAL)

        has_success = processed_overlay is not None or affine_fa_image is not None or affine_dlib_image is not None
        if errors and not has_success:
            messagebox.showerror("特徴点算出エラー", "\n".join(errors))
        elif errors:
            messagebox.showwarning("特徴点算出", "一部成功しました。\n" + "\n".join(errors))

    def save_pipeline_previews(self) -> None:
        if self.current_path is None:
            messagebox.showwarning("プレビュー保存", "先に画像を開いてください。")
            return

        default_dir = self.current_path.parent
        out_dir = filedialog.askdirectory(
            title="プレビュー保存先フォルダ",
            initialdir=str(default_dir),
        )
        if not out_dir:
            return

        stem = self.current_path.stem
        processed_plain = self.pipeline_images.get("processed")
        if processed_plain is None and self.base_image is not None:
            processed_plain = self._get_processed_image()

        save_items = [
            (f"{stem}_01_processed.png", processed_plain),
            (f"{stem}_02_processed_landmarks.png", self.pipeline_images.get("processed_overlay")),
            (f"{stem}_03_affine_face_alignment.png", self.pipeline_images.get("affine_fa")),
            (f"{stem}_04_affine_dlib.png", self.pipeline_images.get("affine_dlib")),
        ]

        saved: list[str] = []
        missing: list[str] = []
        for filename, image in save_items:
            if image is None:
                missing.append(filename)
                continue
            path = Path(out_dir) / filename
            image.save(path, format="PNG")
            saved.append(path.name)

        if self.base_image is not None:
            original_path = Path(out_dir) / f"{stem}_00_original.png"
            self.base_image.save(original_path, format="PNG")
            saved.insert(0, original_path.name)

        if self.cropped_image is not None:
            cropped_path = Path(out_dir) / f"{stem}_00_cropped.png"
            self.cropped_image.save(cropped_path, format="PNG")
            if len(saved) >= 1:
                saved.insert(1, cropped_path.name)
            else:
                saved.insert(0, cropped_path.name)

        if not saved:
            messagebox.showwarning("プレビュー保存", "保存できるプレビューがありません。")
            return

        detail = "\n".join(saved)
        if missing:
            detail += "\n\n未生成のためスキップ:\n" + "\n".join(missing)
        messagebox.showinfo("プレビュー保存", f"保存しました ({len(saved)} 件):\n{detail}")

    def _update_meta(self, array: np.ndarray, path: Path | None = None) -> None:
        lines = []
        if path is not None:
            lines.append(f"file: {path.name}")
            if self.current_file_type == "raw":
                lines.extend(
                    [
                        f"raw width: {self.raw_settings['width']}",
                        f"raw height: {self.raw_settings['height']}",
                        f"raw channels: {self.raw_settings['channels']}",
                        f"raw offset: {self.raw_settings['offset']}",
                        f"raw dtype: {self.raw_settings['dtype']}",
                        f"raw endian: {self.raw_settings['endian']}",
                    ]
                )
        lines.extend(
            [
                f"shape: {array.shape}",
                f"dtype: {array.dtype}",
                f"min: {np.nanmin(array):.6g}",
                f"max: {np.nanmax(array):.6g}",
                f"mean: {np.nanmean(array):.6g}",
                f"std: {np.nanstd(array):.6g}",
            ]
        )
        if array.ndim >= 3:
            lines.append(f"channels: {array.shape[-1]}")

        self.meta_text.config(state=tk.NORMAL)
        self.meta_text.delete("1.0", tk.END)
        self.meta_text.insert(tk.END, "\n".join(lines))
        self.meta_text.config(state=tk.DISABLED)

    @staticmethod
    def array_to_image(array: np.ndarray) -> Image.Image:
        arr = np.asarray(array)

        if arr.ndim == 1:
            side = int(np.sqrt(arr.size))
            if side * side != arr.size:
                raise ValueError("1次元配列は正方形サイズである必要があります。")
            arr = arr.reshape(side, side)

        if arr.ndim > 3:
            arr = np.squeeze(arr)
            if arr.ndim > 3:
                raise ValueError(f"対応していない配列次元です: {array.shape}")

        if arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
                arr = np.moveaxis(arr, 0, -1)
            channels = arr.shape[-1]
            if channels == 1:
                arr = arr[..., 0]
            elif channels >= 3:
                arr = arr[..., :3]

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = arr.astype(np.float64)
        arr -= arr.min()
        max_val = arr.max()
        if max_val > 0:
            arr /= max_val
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        return Image.fromarray(arr, mode="RGB")

    def apply_processing(self, image: Image.Image) -> Image.Image:
        result = image.copy()

        clip_min = self.params["clip_min"].get() / 100.0
        clip_max = self.params["clip_max"].get() / 100.0
        if clip_min > clip_max:
            clip_min, clip_max = clip_max, clip_min

        if clip_min > 0.0 or clip_max < 1.0:
            arr = np.asarray(result, dtype=np.float32) / 255.0
            low = clip_min
            high = max(clip_max, low + 1e-6)
            arr = np.clip((arr - low) / (high - low), 0.0, 1.0)
            result = Image.fromarray((arr * 255).astype(np.uint8), mode=result.mode)

        brightness = self.params["brightness"].get() / 100.0
        contrast = self.params["contrast"].get() / 100.0
        result = ImageEnhance.Brightness(result).enhance(brightness)
        result = ImageEnhance.Contrast(result).enhance(contrast)

        gamma = max(self.params["gamma"].get() / 100.0, 0.01)
        if abs(gamma - 1.0) > 1e-3:
            arr = np.asarray(result, dtype=np.float32) / 255.0
            arr = np.power(arr, gamma)
            result = Image.fromarray((arr * 255).astype(np.uint8), mode=result.mode)

        equalize_strength = self.params["equalize"].get() / 100.0
        if equalize_strength > 0.0:
            equalized = ImageOps.equalize(result.convert("L")).convert(result.mode)
            result = Image.blend(result, equalized, equalize_strength)

        blur_radius = self.params["blur"].get()
        if blur_radius > 0:
            result = result.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        sharpen_amount = self.params["sharpen"].get() / 100.0
        if sharpen_amount > 0.0:
            sharpened = result.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            result = Image.blend(result, sharpened, sharpen_amount)

        threshold = self.params["threshold"].get()
        if threshold > 0:
            gray = result.convert("L")
            binary = gray.point(lambda p: 255 if p >= threshold else 0, mode="L")
            result = binary.convert(result.mode)

        invert_strength = self.params["invert"].get() / 100.0
        if invert_strength > 0.0:
            inverted = ImageOps.invert(result.convert("RGB"))
            if result.mode != "RGB":
                inverted = inverted.convert(result.mode)
            result = Image.blend(result, inverted, invert_strength)

        rotate_angle = self.params["rotate"].get()
        if rotate_angle != 0:
            result = result.rotate(rotate_angle, expand=True, fillcolor=0)

        return result

    def to_photo(self, image: Image.Image) -> ImageTk.PhotoImage:
        preview = image.copy()
        preview.thumbnail((self.PREVIEW_MAX_SIZE, self.PREVIEW_MAX_SIZE), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(preview)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app = ImageProcessingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
