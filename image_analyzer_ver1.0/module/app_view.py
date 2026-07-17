"""表示・リサイズ・ROI/ライン操作関連のミックスイン。"""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image

from module.analysis import (
    circle_mask,
    compute_fft_magnitude,
    polygon_mask,
    rect_mask,
    to_float_gray,
)
from module.constants import DEFAULT_RESIZE_METHOD, RESIZE_METHODS
from module.resize import resize_array
from module.visualization import (
    abs_diff_heatmap,
    array_to_display_image,
    overlay_roi_on_image,
    render_fft_image,
)


class ViewMixin:
    """プレビュー描画・リサイズ・ROI/ラインのインタラクションを担当する。"""

    # ---------- Interaction ----------
    def _sync_interaction_mode(self) -> None:
        tool = self.tool_mode.get()
        if tool == "ライン":
            self.main_canvas.set_interaction_mode("line")
        elif tool == "ROI":
            shape = self.roi_mode.get()
            if shape == "矩形":
                self.main_canvas.set_interaction_mode("rect")
            elif shape == "円":
                self.main_canvas.set_interaction_mode("circle")
            else:
                self.main_canvas.set_interaction_mode("freehand")
        else:
            self.main_canvas.set_interaction_mode("none")

    # ---------- Resize ----------
    def _resize_scale(self) -> float:
        return max(1, min(10, int(self.resize_var.get()))) / 10.0

    def _get_resize_method_name(self) -> str:
        name = self.resize_method_var.get()
        if name in RESIZE_METHODS:
            return name
        return DEFAULT_RESIZE_METHOD

    def _resize_array(self, array: np.ndarray, scale: float) -> np.ndarray:
        return resize_array(array, scale, self._get_resize_method_name())

    def _rebuild_working_array(self) -> None:
        if self.source_array is None:
            self.working_array = None
            return
        self.working_array = self._resize_array(self.source_array, self._resize_scale())

    def _analysis_array(self) -> np.ndarray | None:
        """描画・対話解析に使う配列（リサイズ後）。"""
        return self.working_array if self.working_array is not None else self.source_array

    def _on_resize_slider(self, value: str) -> None:
        snapped = max(1, min(10, int(round(float(value)))))
        if snapped != int(self.resize_var.get()):
            self.resize_var.set(snapped)
        scale = snapped / 10.0
        if self.resize_value_label is not None:
            self.resize_value_label.config(text=f"{scale:.1f}x")
        self._schedule_resize()

    def _on_resize_method_change(self) -> None:
        self._schedule_resize()

    def _schedule_resize(self) -> None:
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(80, self._apply_resize)

    def _apply_resize(self) -> None:
        self._resize_job = None
        if self.source_array is None:
            return
        self.clear_roi(refresh=False)
        self.clear_line(refresh=False)
        self.compare_result = None
        self._rebuild_working_array()
        self.refresh_view(fit=True)
        self.update_stats_panel()
        self.update_histogram()
        self.update_quality()
        self.update_fft()
        if self.compare_array is not None:
            self.run_compare(silent=True)
        arr = self._analysis_array()
        if arr is not None:
            h, w = to_float_gray(arr).shape
            method = self._get_resize_method_name()
            self.status_var.set(f"リサイズ {self._resize_scale():.1f}x [{method}] → {w}×{h}")

    # ---------- View ----------
    def fit_view(self) -> None:
        self.main_canvas.fit_to_window()

    def actual_size_view(self) -> None:
        self.main_canvas.actual_size()

    def adjust_zoom(self, factor: float) -> None:
        self.main_canvas.set_zoom(self.main_canvas.get_zoom() * factor)

    def refresh_view(self, *, fit: bool = False) -> None:
        arr = self._analysis_array()
        if arr is None:
            self.main_canvas.set_image(None)
            return

        mode = self.view_mode.get()
        cmap = self.cmap_var.get()

        if mode == "FFT":
            mag = compute_fft_magnitude(arr)
            display = render_fft_image(mag, cmap=cmap if cmap != "gray" else "inferno")
        elif mode == "差分ヒートマップ":
            if self.compare_result is None:
                if self.compare_array is not None:
                    self.run_compare(silent=True)
                if self.compare_result is None:
                    display = array_to_display_image(arr, cmap=None)
                    self.status_var.set("比較画像が未設定のため通常表示です")
                else:
                    display = abs_diff_heatmap(self.compare_result["abs_diff"], cmap="hot")
            else:
                display = abs_diff_heatmap(self.compare_result["abs_diff"], cmap="hot")
        elif mode == "カラーマップ":
            display = array_to_display_image(arr, cmap=cmap)
        else:
            display = array_to_display_image(arr, cmap=None)

        # 表示解像度を解析配列に合わせる
        gray = to_float_gray(arr)
        if display.size != (gray.shape[1], gray.shape[0]):
            display = display.resize((gray.shape[1], gray.shape[0]), Image.Resampling.NEAREST)

        display = overlay_roi_on_image(
            display,
            rect=self.roi_rect if self._drag_preview is None else None,
            circle=self.roi_circle if self._drag_preview is None else None,
            polygon=self.roi_polygon if self.roi_polygon and self._drag_preview is None else None,
            line=self.line_points if self._drag_preview is None else None,
        )
        if self._drag_preview is not None:
            kind = self._drag_preview[0]
            if kind == "rect":
                display = overlay_roi_on_image(display, rect=self._drag_preview[1])
            elif kind == "circle":
                display = overlay_roi_on_image(display, circle=self._drag_preview[1])
            elif kind == "freehand":
                display = overlay_roi_on_image(display, polygon=list(self._drag_preview[1]))
            elif kind == "line":
                display = overlay_roi_on_image(display, line=self._drag_preview[1])

        self.main_canvas.set_image(display, source_array=arr)
        if fit:
            self.root.update_idletasks()
            self.main_canvas.fit_to_window()

    def _on_main_cursor_move(self, x: int, y: int, _event: tk.Event | None) -> None:
        if x < 0 or self._analysis_array() is None or self.current_path is None:
            return
        val = self.main_canvas.get_pixel_value(x, y)
        self.status_var.set(f"{self.current_path.name}  ({x}, {y}) = {val}")

    # ---------- ROI / line interaction ----------
    def clear_roi(self, *, refresh: bool = True) -> None:
        self.roi_rect = None
        self.roi_circle = None
        self.roi_polygon = []
        self.roi_mask = None
        self._drag_preview = None
        self._set_text(self.roi_text, "ROI 未設定")
        if refresh:
            self.refresh_view()
            self.update_histogram()

    def clear_line(self, *, refresh: bool = True) -> None:
        self.line_points = None
        self._drag_preview = None
        self.profile_info.config(text="ライン未設定")
        self.profile_label.config(image="")
        self._profile_photo = None
        if refresh:
            self.refresh_view()

    def _on_canvas_click(self, x: int, y: int) -> None:
        if self.tool_mode.get() != "ROI" or self.roi_mode.get() != "自由選択":
            return
        self.roi_polygon.append((x, y))
        if len(self.roi_polygon) >= 3:
            self._apply_polygon_roi()
        self.refresh_view()

    def _on_canvas_drag(self, x0: int, y0: int, x1: int, y1: int, phase: str) -> None:
        mode = self.main_canvas.interaction_mode
        if mode == "rect":
            self._drag_preview = ("rect", (x0, y0, x1, y1))
        elif mode == "circle":
            r = float(np.hypot(x1 - x0, y1 - y0))
            self._drag_preview = ("circle", (x0, y0, r))
        elif mode == "line":
            self._drag_preview = ("line", (x0, y0, x1, y1))
        elif mode == "freehand":
            if phase == "start":
                self.roi_polygon = [(x0, y0)]
            self.roi_polygon.append((x1, y1))
            self._drag_preview = ("freehand", list(self.roi_polygon))
        self.refresh_view()

    def _on_canvas_drag_end(self, x0: int, y0: int, x1: int, y1: int) -> None:
        mode = self.main_canvas.interaction_mode
        self._drag_preview = None
        arr = self._analysis_array()
        if arr is None:
            return
        shape = to_float_gray(arr).shape

        if mode == "rect":
            self.roi_rect = (x0, y0, x1, y1)
            self.roi_circle = None
            self.roi_polygon = []
            self.roi_mask = rect_mask(shape, x0, y0, x1, y1)
            self.update_roi_stats()
            self.update_histogram()
        elif mode == "circle":
            r = float(np.hypot(x1 - x0, y1 - y0))
            self.roi_circle = (x0, y0, r)
            self.roi_rect = None
            self.roi_polygon = []
            self.roi_mask = circle_mask(shape, x0, y0, r)
            self.update_roi_stats()
            self.update_histogram()
        elif mode == "line":
            self.line_points = (x0, y0, x1, y1)
            self.update_profile()
        elif mode == "freehand":
            self.roi_polygon.append((x1, y1))
            self._apply_polygon_roi()
        self.refresh_view()

    def _apply_polygon_roi(self) -> None:
        arr = self._analysis_array()
        if arr is None or len(self.roi_polygon) < 3:
            return
        shape = to_float_gray(arr).shape
        self.roi_rect = None
        self.roi_circle = None
        self.roi_mask = polygon_mask(shape, self.roi_polygon)
        self.update_roi_stats()
        self.update_histogram()
