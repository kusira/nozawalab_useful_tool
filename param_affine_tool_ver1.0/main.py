"""NPY/RAW/PNG/JPEG画像の読み込みと画像処理パラメータをスライダーで調整するGUIアプリ。

実装は module/ 以下に機能ごとに分割している。
このファイルはアプリの状態初期化とミックスインの合成、エントリポイントのみを持つ。
"""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.app_crop import CropMixin  # noqa: E402
from module.app_image_io import ImageIOMixin  # noqa: E402
from module.app_landmark import LandmarkMixin  # noqa: E402
from module.app_navigation import NavigationMixin  # noqa: E402
from module.app_preview import PreviewMixin  # noqa: E402
from module.app_ui import UIBuildMixin  # noqa: E402
from module.constants import DEFAULT_RESIZE_METHOD  # noqa: E402
from module.fa_landmark_calculator import resolve_torch_device  # noqa: E402


class ImageProcessingApp(
    UIBuildMixin,
    NavigationMixin,
    ImageIOMixin,
    CropMixin,
    PreviewMixin,
    LandmarkMixin,
):
    """NPY/RAW/PNG/JPEG画像プレビュー・特徴点解析アプリのメイン。

    UI 構築・ナビゲーション・画像入出力・トリミング・プレビュー・特徴点算出の
    各機能は module/app_*.py のミックスインに分割している。
    """

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
        self.root_sources: list[Path] = []
        self.file_list: list[Path] = []
        self.current_index: int = -1
        self._file_tree_iids: dict[int, str] = {}
        self.index_var = tk.StringVar(value="")
        self.resize_method_var = tk.StringVar(value=DEFAULT_RESIZE_METHOD)
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
        self._bind_shortcuts()


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app = ImageProcessingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
