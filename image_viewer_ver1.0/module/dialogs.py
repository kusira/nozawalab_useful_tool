"""ダイアログウィンドウ（RAW読み込み設定など）。"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np

from module.image_loader import RAW_DTYPE_OPTIONS


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
