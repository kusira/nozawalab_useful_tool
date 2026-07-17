"""ディレクトリ／単体ファイル対応の多機能画像ビューワー。

実装は module/ 以下に機能ごとに分割している。
このファイルはアプリの状態初期化とミックスインの合成、エントリポイントのみを持つ。
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.app_export import ExportMixin  # noqa: E402
from module.app_navigation import NavigationMixin  # noqa: E402
from module.app_ui import UIBuildMixin  # noqa: E402
from module.app_view import ViewMixin  # noqa: E402
from module.canvas import MagnifierWindow  # noqa: E402
from module.constants import default_downloads_dir  # noqa: E402
from module.file_tree import directory_has_images  # noqa: E402
from module.image_loader import DEFAULT_RAW_SETTINGS, is_supported_image  # noqa: E402
from module.image_processing import DEFAULT_RESIZE_METHOD, default_params  # noqa: E402


class ImageViewerApp(
    UIBuildMixin,
    NavigationMixin,
    ViewMixin,
    ExportMixin,
):
    """多機能画像ビューワーのメインアプリケーション。

    UI 構築・ナビゲーション・表示・エクスポートの各機能は
    module/app_*.py のミックスインに分割している。
    """

    BROWSE_MODES = ("順番", "ランダム", "番号指定")

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
