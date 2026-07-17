"""画像保存・バッチ出力のミックスイン。"""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

from module.analysis import (
    assess_quality,
    compute_fft_magnitude,
    compute_histogram,
    extract_line_profile,
    stats_row_for_export,
)
from module.image_loader import guess_raw_settings, load_image_file
from module.visualization import (
    render_fft_image,
    render_histogram_image,
    render_profile_image,
)


class ExportMixin:
    """解析結果の保存とバッチ出力を担当する。"""

    def choose_export_dir(self) -> None:
        path = filedialog.askdirectory(title="保存先フォルダを選択", initialdir=str(self.export_dir))
        if path:
            self.export_dir = Path(path)
            self.status_var.set(f"保存先: {self.export_dir}")

    def _unique_path(self, directory: Path, stem: str, suffix: str) -> Path:
        candidate = directory / f"{stem}{suffix}"
        if not candidate.exists():
            return candidate
        i = 1
        while True:
            candidate = directory / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                return candidate
            i += 1

    def save_display_image(self) -> None:
        img = self.main_canvas.get_image()
        if img is None:
            return
        stem = (self.current_path.stem if self.current_path else "display") + "_analyzed"
        path = filedialog.asksaveasfilename(
            title="解析画像を保存",
            initialdir=str(self.export_dir),
            initialfile=f"{stem}.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if not path:
            return
        img.save(path)
        self.status_var.set(f"保存しました: {path}")

    def save_histogram_image(self) -> None:
        # 自動計算オフで未表示でも、保存時はその場で計算して保存する。
        arr = self._analysis_array()
        if arr is None:
            return
        hist = compute_histogram(
            arr,
            bins=max(16, int(self.hist_bins.get())),
            mask=self.roi_mask,
            channel=self.hist_channel.get(),
            value_range=self._get_hist_range(),
            auto_bins=True,
        )
        img = render_histogram_image(hist, show_cdf=self.show_cdf.get())
        stem = (self.current_path.stem if self.current_path else "hist") + "_histogram"
        out = self._unique_path(self.export_dir, stem, ".png")
        img.save(out)
        self.status_var.set(f"保存しました: {out}")

    def save_profile_image(self) -> None:
        arr = self._analysis_array()
        if arr is None or self.line_points is None:
            messagebox.showinfo("プロファイル", "ラインが未設定です。")
            return
        x0, y0, x1, y1 = self.line_points
        profile = extract_line_profile(arr, x0, y0, x1, y1)
        img = render_profile_image(profile["distance"], profile["values"])
        stem = (self.current_path.stem if self.current_path else "profile") + "_profile"
        out = self._unique_path(self.export_dir, stem, ".png")
        img.save(out)
        self.status_var.set(f"保存しました: {out}")

    def save_fft_image(self) -> None:
        arr = self._analysis_array()
        if arr is None:
            return
        mag = compute_fft_magnitude(arr)
        img = render_fft_image(mag, cmap=self.cmap_var.get() if self.cmap_var.get() != "gray" else "inferno")
        stem = (self.current_path.stem if self.current_path else "fft") + "_fft"
        out = self._unique_path(self.export_dir, stem, ".png")
        img.save(out)
        self.status_var.set(f"保存しました: {out}")

    def run_batch_export(self, fmt: str) -> None:
        if not self.file_list:
            messagebox.showinfo("バッチ", "ファイル一覧が空です。")
            return
        if self._batch_busy:
            messagebox.showinfo("バッチ", "処理中です。")
            return

        if fmt == "csv":
            path = filedialog.asksaveasfilename(
                title="CSVの保存先",
                initialdir=str(self.export_dir),
                initialfile="batch_analysis.csv",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")],
            )
        else:
            path = filedialog.asksaveasfilename(
                title="JSONの保存先",
                initialdir=str(self.export_dir),
                initialfile="batch_analysis.json",
                defaultextension=".json",
                filetypes=[("JSON", "*.json")],
            )
        if not path:
            return

        include_quality = self.batch_include_quality.get()
        targets = list(self.file_list)
        raw_settings = dict(self.raw_settings)

        def worker() -> None:
            rows: list[dict] = []
            errors: list[str] = []
            for i, fp in enumerate(targets):
                self.root.after(
                    0,
                    lambda i=i, n=len(targets), name=fp.name: self.batch_status.config(
                        text=f"処理中... {i + 1}/{n}  {name}"
                    ),
                )
                try:
                    settings = raw_settings if fp.suffix.lower() == ".raw" else None
                    if settings is not None:
                        settings = guess_raw_settings(fp, raw_settings)
                    array, _img, _t = load_image_file(fp, settings)
                    quality = assess_quality(array) if include_quality else None
                    rows.append(stats_row_for_export(fp, array, quality))
                except Exception as exc:
                    errors.append(f"{fp.name}: {exc}")

            try:
                out = Path(path)
                if fmt == "csv":
                    if rows:
                        fieldnames = list(rows[0].keys())
                        with out.open("w", newline="", encoding="utf-8-sig") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)
                else:
                    with out.open("w", encoding="utf-8") as f:
                        json.dump({"rows": rows, "errors": errors}, f, ensure_ascii=False, indent=2)
                msg = f"完了: {len(rows)} 件 → {out}"
                if errors:
                    msg += f"（失敗 {len(errors)} 件）"
                self.root.after(0, lambda: self._batch_done(msg, errors))
            except Exception as exc:
                self.root.after(0, lambda: self._batch_done(f"保存失敗: {exc}", errors))

        self._batch_busy = True
        self.batch_status.config(text="バッチ処理を開始しています...")
        threading.Thread(target=worker, daemon=True).start()

    def _batch_done(self, message: str, errors: list[str]) -> None:
        self._batch_busy = False
        self.batch_status.config(text=message)
        self.status_var.set(message)
        if errors:
            messagebox.showwarning("バッチ", message + "\n\n" + "\n".join(errors[:10]))
        else:
            messagebox.showinfo("バッチ", message)
