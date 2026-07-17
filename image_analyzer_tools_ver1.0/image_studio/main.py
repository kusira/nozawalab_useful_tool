"""image_studio — 画像ビューワー + 画像解析の統合アプリ。

ディレクトリ／単体ファイルから画像を読み込み、表示・画像処理・ルーペ・
統計/ヒストグラム/ROI/プロファイル/比較/品質/FFT/バッチ解析・エクスポートを行う。

実装は module/ 以下に機能ごとに分割し、共通処理は上位の common/ を参照する。
このファイルはアプリの状態初期化とミックスインの合成、エントリポイントのみを持つ。
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

APP_ROOT = Path(__file__).resolve().parent
SUITE_ROOT = APP_ROOT.parent
for _p in (str(APP_ROOT), str(SUITE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.constants import DEFAULT_RESIZE_METHOD, default_downloads_dir  # noqa: E402
from common.file_tree import directory_has_images  # noqa: E402
from common.image_loader import DEFAULT_RAW_SETTINGS, is_supported_image  # noqa: E402
from module.app_analysis import AnalysisMixin  # noqa: E402
from module.app_export import ExportMixin  # noqa: E402
from module.app_navigation import NavigationMixin  # noqa: E402
from module.app_ui import UIBuildMixin  # noqa: E402
from module.app_view import ViewMixin  # noqa: E402
from module.canvas import MagnifierWindow  # noqa: E402
from module.image_processing import default_params  # noqa: E402


class ImageStudioApp(
    UIBuildMixin,
    NavigationMixin,
    ViewMixin,
    AnalysisMixin,
    ExportMixin,
):
    """ビューワー機能と解析機能を統合したメインアプリケーション。"""

    BROWSE_MODES = ("順番", "ランダム", "番号指定")
    VIEW_MODES = ("通常", "カラーマップ", "FFT", "差分ヒートマップ")

    def __init__(self, root: tk.Tk, initial_paths: list[Path] | None = None) -> None:
        self.root = root
        self.root.title("image_studio — 画像ビューワー・解析")
        self.root.minsize(1400, 900)

        # ---------- 状態 ----------
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

        # ---------- Tk 変数 ----------
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
        self.hist_auto = tk.BooleanVar(value=True)
        self.hist_range_min = tk.StringVar(value="")
        self.hist_range_max = tk.StringVar(value="")
        self.roi_mode = tk.StringVar(value="矩形")
        self.tool_mode = tk.StringVar(value="なし")  # なし / ROI / ライン
        self._resize_job: str | None = None
        self._view_job: str | None = None
        self.resize_value_label: ttk.Label | None = None

        # ---------- 画像処理（表示調整）----------
        self.param_values: dict[str, int] = default_params()
        self.param_vars: dict[str, tk.IntVar] = {}
        self.param_labels: dict[str, ttk.Label] = {}
        self.export_dir_label: ttk.Label | None = None

        # ---------- ルーペ ----------
        self.magnifier = MagnifierWindow(self.root)
        self.magnifier_enabled = tk.BooleanVar(value=True)
        self.magnifier_size = tk.IntVar(value=64)
        self.magnifier_zoom = tk.IntVar(value=2)
        self.magnifier_radius_label: ttk.Label | None = None
        self.magnifier_zoom_label: ttk.Label | None = None

        # ---------- ROI / ライン ----------
        self.roi_rect: tuple[int, int, int, int] | None = None
        self.roi_circle: tuple[int, int, float] | None = None
        self.roi_polygon: list[tuple[int, int]] = []
        self.line_points: tuple[int, int, int, int] | None = None
        self.roi_mask: np.ndarray | None = None

        # ---------- 比較 ----------
        self.compare_array: np.ndarray | None = None
        self.compare_path: Path | None = None
        self.compare_result: dict | None = None
        self._drag_preview: tuple | None = None

        # ---------- PhotoImage 参照保持 ----------
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
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    initial = parse_cli_paths(sys.argv[1:])
    ImageStudioApp(root, initial_paths=initial or None)
    root.mainloop()


if __name__ == "__main__":
    main()
