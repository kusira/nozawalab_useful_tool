"""処理後画像のエクスポート関連のミックスイン。"""

from __future__ import annotations

from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image


class ExportMixin:
    """処理後画像の保存先管理とエクスポートを担当する。"""

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
