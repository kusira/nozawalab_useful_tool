"""画像解析ツール — 統計・ヒストグラム・ROI・プロファイル・比較・品質・FFT・バッチ。

実装は module/ 以下に機能ごとに分割している。
このファイルはアプリの状態初期化とミックスインの合成、エントリポイントのみを持つ。
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.app_analysis import AnalysisMixin  # noqa: E402
from module.app_export import ExportMixin  # noqa: E402
from module.app_navigation import NavigationMixin  # noqa: E402
from module.app_ui import UIBuildMixin  # noqa: E402
from module.app_view import ViewMixin  # noqa: E402
from module.constants import DEFAULT_RESIZE_METHOD, default_downloads_dir  # noqa: E402
from module.image_loader import DEFAULT_RAW_SETTINGS  # noqa: E402


class ImageAnalyzerApp(
    UIBuildMixin,
    NavigationMixin,
    ViewMixin,
    AnalysisMixin,
    ExportMixin,
):
    """画像解析ツールのメインアプリケーション。

    UI 構築・ナビゲーション・表示・解析・エクスポートの各機能は
    module/app_*.py のミックスインに分割している。
    """

    BROWSE_MODES = ("順番", "ランダム", "番号指定")
    VIEW_MODES = ("通常", "カラーマップ", "FFT", "差分ヒートマップ")

    def __init__(self, root: tk.Tk, initial_paths: list[Path] | None = None) -> None:
        self.root = root
        self.root.title("画像解析ツール")
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
        # bins は常に画素の階調に合わせて自動調整（フォールバック用の既定値）
        self.hist_bins = tk.IntVar(value=256)
        self.show_cdf = tk.BooleanVar(value=True)
        # 画像切替時のヒストグラム自動計算（重いのでオフにできる。オフ時は「更新」で計算）
        self.hist_auto = tk.BooleanVar(value=True)
        # ヒストグラムのカウント対象とする値域（空欄で全体）
        self.hist_range_min = tk.StringVar(value="")
        self.hist_range_max = tk.StringVar(value="")
        self.roi_mode = tk.StringVar(value="矩形")
        self.tool_mode = tk.StringVar(value="なし")  # なし / ROI / ライン
        self._resize_job: str | None = None
        self.resize_value_label: ttk.Label | None = None

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
