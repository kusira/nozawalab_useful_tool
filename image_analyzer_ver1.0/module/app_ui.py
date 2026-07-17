"""ウィジェット構築（メニュー・ツールバー・各パネル/タブ）のミックスイン。"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from module.analysis import COLORMAPS
from module.canvas import ZoomableCanvas
from module.constants import RESIZE_METHODS


class UIBuildMixin:
    """アプリのウィジェットを組み立てる。"""

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
        ttk.Checkbutton(row, text="CDF", variable=self.show_cdf).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(
            row,
            text="自動計算",
            variable=self.hist_auto,
            command=self._on_hist_auto_toggle,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(row, text="更新", command=lambda: self.update_histogram(force=True)).pack(side=tk.RIGHT)

        range_row = ttk.Frame(parent)
        range_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(range_row, text="値域").pack(side=tk.LEFT)
        ttk.Label(range_row, text="min").pack(side=tk.LEFT, padx=(6, 2))
        min_entry = ttk.Entry(range_row, textvariable=self.hist_range_min, width=8)
        min_entry.pack(side=tk.LEFT)
        ttk.Label(range_row, text="max").pack(side=tk.LEFT, padx=(6, 2))
        max_entry = ttk.Entry(range_row, textvariable=self.hist_range_max, width=8)
        max_entry.pack(side=tk.LEFT)
        min_entry.bind("<Return>", lambda _e: self.update_histogram(force=True))
        max_entry.bind("<Return>", lambda _e: self.update_histogram(force=True))
        ttk.Button(range_row, text="全体", command=self._clear_hist_range).pack(side=tk.RIGHT)
        ttk.Button(range_row, text="データ範囲", command=self._fill_hist_range_from_data).pack(
            side=tk.RIGHT, padx=(0, 4)
        )
        ttk.Label(
            parent,
            text="bins は画素の階調に合わせて自動調整されます。空欄で全体を集計。範囲を指定すると黒つぶれ等の偏りを除外できます。",
            foreground="#888888",
            wraplength=380,
        ).pack(anchor=tk.W, pady=(2, 0))

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
