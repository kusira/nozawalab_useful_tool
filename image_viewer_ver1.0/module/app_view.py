"""表示更新・画像処理適用・ルーペ・メタ情報のミックスイン。"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from module.image_loader import array_stats
from module.image_processing import DEFAULT_RESIZE_METHOD, apply_processing, default_params


class ViewMixin:
    """プレビュー描画・パラメータ適用・ルーペ表示を担当する。"""

    def schedule_refresh(self) -> None:
        if self._update_job is not None:
            self.root.after_cancel(self._update_job)
        self._update_job = self.root.after(60, self._run_refresh)

    def _run_refresh(self) -> None:
        self._update_job = None
        self.refresh_view()

    def _get_processed_image(self) -> Image.Image | None:
        if self.base_image is None:
            return None
        return apply_processing(
            self.base_image,
            self.param_values,
            resize_method=self._get_resize_resampling(),
        )

    def refresh_view(self) -> None:
        processed = self._get_processed_image()
        if processed is None:
            self.main_canvas.set_image(None)
            return
        self.main_canvas.set_resampling(self._get_resize_resampling())
        self.main_canvas.set_image(processed, self.source_array)

    def reset_params(self) -> None:
        defaults = default_params()
        for key, value in defaults.items():
            self.param_values[key] = value
            if key in self.param_vars:
                self.param_vars[key].set(value)
                self.param_labels[key].config(text=self._format_param_value(key, value))
        self.resize_method_var.set(DEFAULT_RESIZE_METHOD)
        self.refresh_view()

    def flip_horizontal(self) -> None:
        if self.base_image is None:
            return
        self.base_image = ImageOps.mirror(self.base_image)
        if self.source_array is not None:
            self.source_array = np.flip(self.source_array, axis=1)
        self.refresh_view()

    def flip_vertical(self) -> None:
        if self.base_image is None:
            return
        self.base_image = ImageOps.flip(self.base_image)
        if self.source_array is not None:
            self.source_array = np.flip(self.source_array, axis=0)
        self.refresh_view()

    def rotate_by(self, angle: int) -> None:
        current = self.param_values.get("rotate", 0)
        new_angle = max(-180, min(180, current + angle))
        self.param_values["rotate"] = new_angle
        if "rotate" in self.param_vars:
            self.param_vars["rotate"].set(new_angle)
            self.param_labels["rotate"].config(text=str(new_angle))
        self.refresh_view()

    def fit_view(self) -> None:
        self.main_canvas.fit_to_window()

    def actual_size_view(self) -> None:
        self.main_canvas.actual_size()

    def adjust_zoom(self, factor: float) -> None:
        self.main_canvas.set_zoom(self.main_canvas.get_zoom() * factor)

    def _on_magnifier_radius_change(self, val: str) -> None:
        value = int(float(val))
        self.magnifier_size.set(value)
        if self.magnifier_radius_label is not None:
            self.magnifier_radius_label.config(text=str(value))

    def _on_magnifier_zoom_change(self, val: str) -> None:
        value = int(float(val))
        self.magnifier_zoom.set(value)
        if self.magnifier_zoom_label is not None:
            self.magnifier_zoom_label.config(text=f"{value}x")

    def _update_meta(self, path: Path, array: np.ndarray) -> None:
        stats = array_stats(array)
        lines = [
            f"path: {path.name}",
            f"type: {self.current_file_type}",
            f"shape: {stats['shape']}",
            f"dtype: {stats['dtype']}",
            f"min: {stats['min']:.4g}",
            f"max: {stats['max']:.4g}",
            f"mean: {stats['mean']:.4g}",
            f"std: {stats['std']:.4g}",
            f"size: {path.stat().st_size:,} bytes",
            f"no: {self.format_list_number(self.current_index)}",
            f"index: {self.current_index + 1}/{len(self.file_list)}",
        ]
        self.meta_text.config(state=tk.NORMAL)
        self.meta_text.delete("1.0", tk.END)
        self.meta_text.insert(tk.END, "\n".join(lines))
        self.meta_text.config(state=tk.DISABLED)

    def _on_main_cursor_move(self, x: int, y: int, event: tk.Event | None) -> None:
        if x < 0 or y < 0:
            self.magnifier.hide()
            self.status_var.set("準備完了")
            return

        pixel = self.main_canvas.get_pixel_value(x, y)
        zoom = self.main_canvas.get_zoom()
        self.status_var.set(f"({x}, {y})  value={pixel}  zoom={zoom:.2f}x")

        if not self.magnifier_enabled.get() or event is None:
            self.magnifier.hide()
            return

        patch = self.main_canvas.extract_patch(
            x, y, self.magnifier_size.get(), self.magnifier_zoom.get()
        )
        if patch is None:
            return
        sx = event.x_root
        sy = event.y_root
        self.magnifier.show_patch(patch, sx, sy)
