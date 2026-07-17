"""ウィジェット構築（ツールバー・ファイルパネル・コントロール・プレビュー）のミックスイン。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from common.constants import DEFAULT_RESIZE_METHOD, RESIZE_METHODS
from module.face_pipeline import AFFINE_SIZE_HEIGHT, AFFINE_SIZE_WIDTH


class UIBuildMixin:
    """アプリのウィジェットを組み立て、スライダーの値変換を担当する。"""

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Button(top, text="ファイルを開く", command=self.open_files).pack(side=tk.LEFT)
        ttk.Button(top, text="フォルダを開く", command=self.open_directory).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(top, text="追加", command=self.add_files).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(top, text="◀", width=3, command=self.show_prev).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="▶", width=3, command=self.show_next).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Label(top, text="番号").pack(side=tk.LEFT, padx=(8, 2))
        index_entry = ttk.Entry(top, textvariable=self.index_var, width=6)
        index_entry.pack(side=tk.LEFT)
        index_entry.bind("<Return>", lambda _e: self.jump_to_index())
        ttk.Button(top, text="移動", command=self.jump_to_index).pack(side=tk.LEFT, padx=(2, 0))

        self.path_label = ttk.Label(top, text="ファイル未選択")
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

        file_panel = ttk.Frame(main, padding=4, width=260)
        main.add(file_panel, weight=0)

        controls = ttk.Frame(main, padding=8, width=320)
        main.add(controls, weight=0)

        preview = ttk.Frame(main, padding=8)
        main.add(preview, weight=1)

        self._build_file_panel(file_panel)
        self._build_controls(controls)
        self._build_preview(preview)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Left>", lambda _e: self.show_prev())
        self.root.bind("<Right>", lambda _e: self.show_next())
        self.root.bind("<Control-o>", lambda _e: self.open_files())
        self.root.bind("<Control-O>", lambda _e: self.open_directory())

    def _build_file_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="ファイル一覧", font=("", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(
            parent,
            text="D0001=フォルダ(青)  0001=ファイル(緑)",
            foreground="#555555",
            font=("", 8),
        ).pack(anchor=tk.W, pady=(2, 0))

        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        self.file_tree = ttk.Treeview(list_frame, show="tree", selectmode="browse")
        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=scroll.set)
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_tree.tag_configure("dir", foreground="#1565c0")
        self.file_tree.tag_configure("file", foreground="#2e7d32")
        self.file_tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.file_tree.bind("<Double-1>", self._on_tree_activate)

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

        method_row = ttk.Frame(preprocess)
        method_row.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(method_row, text="手法", width=12).pack(side=tk.LEFT)
        method_combo = ttk.Combobox(
            method_row,
            textvariable=self.resize_method_var,
            values=list(RESIZE_METHODS.keys()),
            state="readonly",
            width=22,
        )
        method_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        method_combo.bind("<<ComboboxSelected>>", lambda _e: self.schedule_update())

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
