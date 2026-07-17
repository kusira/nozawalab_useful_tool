"""ウィジェット構築（メニュー・ツールバー・各パネル）のミックスイン。"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image

from module.canvas import ZoomableCanvas
from module.image_processing import DEFAULT_RESIZE_METHOD, RESIZE_METHODS


class UIBuildMixin:
    """アプリのウィジェットを組み立てる。"""

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
