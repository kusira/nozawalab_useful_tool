"""統計・ヒストグラム・プロファイル・品質・FFT・比較のミックスイン。"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

from module.analysis import (
    assess_quality,
    compare_images,
    compute_fft_magnitude,
    compute_histogram,
    compute_stats,
    extract_line_profile,
    to_float_gray,
)
from module.dialogs import RawSettingsDialog
from module.image_loader import guess_raw_settings, load_image_file
from module.visualization import (
    abs_diff_heatmap,
    render_fft_image,
    render_histogram_image,
    render_profile_image,
)


class AnalysisMixin:
    """右パネル各タブの解析結果を更新する。"""

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.config(state=tk.DISABLED)

    @staticmethod
    def _format_stats(st: dict, title: str = "統計") -> str:
        lines = [
            f"[{title}]",
            f"shape   : {st['shape']}",
            f"dtype   : {st['dtype']}",
            f"count   : {st['count']}",
            f"min     : {st['min']:.6g}",
            f"max     : {st['max']:.6g}",
            f"mean    : {st['mean']:.6g}",
            f"std     : {st['std']:.6g}",
            f"median  : {st['median']:.6g}",
            f"p01/p05 : {st['p01']:.6g} / {st['p05']:.6g}",
            f"p25/p75 : {st['p25']:.6g} / {st['p75']:.6g}",
            f"p95/p99 : {st['p95']:.6g} / {st['p99']:.6g}",
            f"zero%   : {st['zero_ratio'] * 100:.2f}",
            f"satur%  : {st['saturated_ratio'] * 100:.2f}",
            f"NaN/Inf : {st['nan_count']} / {st['inf_count']}",
        ]
        return "\n".join(lines)

    def update_stats_panel(self) -> None:
        arr = self._analysis_array()
        if arr is None or self.source_array is None:
            self._set_text(self.meta_text, "")
            return
        st = compute_stats(arr)
        scale = self._resize_scale()
        src_h, src_w = to_float_gray(self.source_array).shape
        work_h, work_w = to_float_gray(arr).shape
        header = []
        if self.current_path is not None:
            header.append(f"file: {self.current_path.name}")
        header.append(f"resize : {scale:.1f}x ({self._get_resize_method_name()})")
        header.append(f"src    : {src_w}×{src_h}")
        header.append(f"work   : {work_w}×{work_h}")
        text = "\n".join(header) + "\n\n" + self._format_stats(st, "解析対象（リサイズ後）")
        self._set_text(self.meta_text, text)

    # ---------- Histogram ----------
    def _get_hist_range(self) -> tuple[float, float] | None:
        """入力欄からヒストグラムの値域を取得する。空欄・不正時は None（全体）。"""
        min_text = self.hist_range_min.get().strip()
        max_text = self.hist_range_max.get().strip()
        if not min_text and not max_text:
            return None
        arr = self._analysis_array()
        data_lo, data_hi = 0.0, 1.0
        if arr is not None:
            gray = to_float_gray(arr)
            finite = gray[np.isfinite(gray)]
            if finite.size:
                data_lo, data_hi = float(finite.min()), float(finite.max())
        try:
            lo = float(min_text) if min_text else data_lo
            hi = float(max_text) if max_text else data_hi
        except ValueError:
            messagebox.showwarning("値域", "min / max には数値を入力してください。")
            return None
        if hi < lo:
            lo, hi = hi, lo
        if hi == lo:
            hi = lo + 1e-6
        return (lo, hi)

    def _clear_hist_range(self) -> None:
        self.hist_range_min.set("")
        self.hist_range_max.set("")
        self.update_histogram(force=True)

    def _fill_hist_range_from_data(self) -> None:
        arr = self._analysis_array()
        if arr is None:
            return
        gray = to_float_gray(arr)
        finite = gray[np.isfinite(gray)]
        if not finite.size:
            return
        # 画素値は整数扱い。全データを含むよう min は切り捨て、max は切り上げ。
        lo = int(np.floor(float(finite.min())))
        hi = int(np.ceil(float(finite.max())))
        self.hist_range_min.set(str(lo))
        self.hist_range_max.set(str(hi))
        self.update_histogram(force=True)

    def _on_hist_auto_toggle(self) -> None:
        """自動計算のオン/オフ切替時の処理。"""
        if self.hist_auto.get():
            # オンにしたら現在の画像で即計算する。
            self.update_histogram(force=True)
        else:
            # オフにしたら表示を消して案内を出す。
            self._plot_photo = None
            self.hist_label.config(image="", text="自動計算オフ：「更新」で計算します。")
            self.status_var.set("ヒストグラム自動計算：オフ")

    def update_histogram(self, *, force: bool = False) -> None:
        arr = self._analysis_array()
        if arr is None:
            return
        # 画像切替などの自動呼び出しは、自動計算オフのときスキップする（重い処理の回避）。
        if not force and not self.hist_auto.get():
            self._plot_photo = None
            self.hist_label.config(image="", text="自動計算オフ：「更新」で計算します。")
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
        self._plot_photo = ImageTk.PhotoImage(img)
        self.hist_label.config(image=self._plot_photo, text="")

    def update_roi_stats(self) -> None:
        arr = self._analysis_array()
        if arr is None or self.roi_mask is None or not self.roi_mask.any():
            self._set_text(self.roi_text, "有効な ROI がありません。")
            return
        st = compute_stats(arr, mask=self.roi_mask)
        info = []
        if self.roi_rect is not None:
            info.append(f"矩形: {self.roi_rect}")
        if self.roi_circle is not None:
            info.append(f"円: center=({self.roi_circle[0]}, {self.roi_circle[1]}) r={self.roi_circle[2]:.1f}")
        if self.roi_polygon:
            info.append(f"多角形: 頂点数={len(self.roi_polygon)}")
        text = "\n".join(info) + "\n\n" + self._format_stats(st, "ROI")
        self._set_text(self.roi_text, text)

    def update_profile(self) -> None:
        arr = self._analysis_array()
        if arr is None or self.line_points is None:
            return
        x0, y0, x1, y1 = self.line_points
        profile = extract_line_profile(arr, x0, y0, x1, y1)
        img = render_profile_image(profile["distance"], profile["values"])
        self._profile_photo = ImageTk.PhotoImage(img)
        self.profile_label.config(image=self._profile_photo)
        vals = profile["values"]
        self.profile_info.config(
            text=(
                f"({x0},{y0})→({x1},{y1})  "
                f"len={profile['distance'][-1]:.1f}px  "
                f"min={vals.min():.4g} max={vals.max():.4g} mean={vals.mean():.4g}"
            )
        )

    def update_quality(self) -> None:
        arr = self._analysis_array()
        if arr is None:
            return
        q = assess_quality(arr)
        lines = [
            "[品質チェック]",
            f"blur_score (Laplacian var): {q.blur_score:.2f}  {'⚠' if q.blur_flag else 'OK'}",
            f"underexposed ratio        : {q.underexposed_ratio * 100:.2f}%  {'⚠' if q.underexposed_flag else 'OK'}",
            f"overexposed ratio         : {q.overexposed_ratio * 100:.2f}%  {'⚠' if q.overexposed_flag else 'OK'}",
            f"noise estimate            : {q.noise_estimate:.2f}  {'⚠' if q.noise_flag else 'OK'}",
            f"resolution OK             : {q.resolution_ok}",
            "",
            "[メモ]",
            *q.notes,
        ]
        self._set_text(self.quality_text, "\n".join(lines))

    def update_fft(self) -> None:
        arr = self._analysis_array()
        if arr is None:
            return
        mag = compute_fft_magnitude(arr)
        # サムネイル表示用に縮小
        img = render_fft_image(mag, cmap=self.cmap_var.get() if self.cmap_var.get() != "gray" else "inferno")
        thumb = img.copy()
        thumb.thumbnail((400, 400), Image.Resampling.BILINEAR)
        self._fft_photo = ImageTk.PhotoImage(thumb)
        self.fft_label.config(image=self._fft_photo)

    # ---------- Compare ----------
    def open_compare_image(self) -> None:
        path = filedialog.askopenfilename(
            title="比較する画像を選択",
            filetypes=[("対応形式", "*.npy *.raw *.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("すべて", "*.*")],
        )
        if not path:
            return
        p = Path(path)
        try:
            raw_settings = None
            if p.suffix.lower() == ".raw":
                guessed = guess_raw_settings(p, self.raw_settings)
                dialog = RawSettingsDialog(self.root, p, guessed)
                self.root.wait_window(dialog)
                if dialog.result is None:
                    return
                raw_settings = dict(dialog.result)
            array, _image, _ftype = load_image_file(p, raw_settings)
            self.compare_array = array
            self.compare_path = p
            self.compare_path_label.config(text=f"比較画像: {p.name}")
            self.run_compare()
        except Exception as exc:
            messagebox.showerror("比較画像", str(exc))

    def run_compare(self, *, silent: bool = False) -> None:
        arr = self._analysis_array()
        if arr is None or self.compare_array is None:
            if not silent:
                messagebox.showinfo("比較", "現在画像と比較画像の両方が必要です。")
            return
        # 比較画像も同じリサイズ率で揃える
        compare_resized = self._resize_array(self.compare_array, self._resize_scale())
        result = compare_images(arr, compare_resized)
        self.compare_result = result
        psnr_val = result["psnr"]
        psnr_text = "inf" if np.isinf(psnr_val) else f"{psnr_val:.4f}"
        text = "\n".join(
            [
                f"比較先: {self.compare_path.name if self.compare_path else '-'}",
                f"比較サイズ: {result['shape']}",
                f"リサイズ: {self._resize_scale():.1f}x",
                f"MSE  : {result['mse']:.6g}",
                f"MAE  : {result['mae']:.6g}",
                f"PSNR : {psnr_text} dB",
                f"SSIM : {result['ssim']:.6f}",
            ]
        )
        self._set_text(self.compare_text, text)
        heat = abs_diff_heatmap(result["abs_diff"], cmap="hot")
        heat.thumbnail((400, 320), Image.Resampling.BILINEAR)
        self._compare_photo = ImageTk.PhotoImage(heat)
        self.compare_label.config(image=self._compare_photo)
        if self.view_mode.get() == "差分ヒートマップ":
            self.refresh_view()
