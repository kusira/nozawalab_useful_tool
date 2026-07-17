"""前処理（反転・トリミング）と元画像キャンバス描画のミックスイン。"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageOps, ImageTk


class CropMixin:
    """左右/上下反転・範囲選択トリミング・元画像キャンバスの描画を担当する。"""

    def flip_original_horizontal(self) -> None:
        if self.base_image is None:
            return
        self.pending_crop = None
        self.cropped_image = None
        self.cropped_array = None
        self.base_image = ImageOps.mirror(self.base_image)
        if self.source_array is not None:
            self.source_array = np.flip(self.source_array, axis=1)
        if self.crop_mode:
            self.crop_apply_button.config(state=tk.DISABLED)
            self._render_original_canvas(self.base_image)
            self._set_preview_frame_title("original", self.base_image)
        else:
            self._update_crop_reset_button_state()
            self.update_preview()

    def flip_original_vertical(self) -> None:
        if self.base_image is None:
            return
        self.pending_crop = None
        self.cropped_image = None
        self.cropped_array = None
        self.base_image = ImageOps.flip(self.base_image)
        if self.source_array is not None:
            self.source_array = np.flip(self.source_array, axis=0)
        if self.crop_mode:
            self.crop_apply_button.config(state=tk.DISABLED)
            self._render_original_canvas(self.base_image)
            self._set_preview_frame_title("original", self.base_image)
        else:
            self._update_crop_reset_button_state()
            self.update_preview()

    def start_crop_mode(self) -> None:
        if self.base_image is None:
            messagebox.showwarning("トリミング", "先に画像を開いてください。")
            return
        self.crop_mode = True
        self.pending_crop = None
        self._crop_drag_start = None
        self.crop_select_button.config(state=tk.DISABLED)
        self.crop_apply_button.config(state=tk.DISABLED)
        self.crop_cancel_button.config(state=tk.NORMAL)
        if self.original_canvas is not None:
            self.original_canvas.config(cursor="crosshair")
        self._bind_crop_events()
        self._render_original_canvas(self.base_image)
        self._set_preview_frame_title("original", self.base_image)

    def cancel_crop_mode(self) -> None:
        was_crop_mode = self.crop_mode
        self.crop_mode = False
        self.pending_crop = None
        self._crop_drag_start = None
        self._unbind_crop_events()
        if self.original_canvas is not None:
            self.original_canvas.config(cursor="arrow")
        self.crop_apply_button.config(state=tk.DISABLED)
        self.crop_cancel_button.config(state=tk.DISABLED)
        if self.base_image is not None:
            self.crop_select_button.config(state=tk.NORMAL)
        else:
            self.crop_select_button.config(state=tk.DISABLED)
        if was_crop_mode and self.base_image is not None:
            self.update_preview()

    def _update_crop_reset_button_state(self) -> None:
        if self.cropped_image is not None:
            self.crop_reset_button.config(state=tk.NORMAL)
        else:
            self.crop_reset_button.config(state=tk.DISABLED)

    def reset_crop(self) -> None:
        if self.base_image is None:
            return
        if self.cropped_image is None and not self.crop_mode and self.pending_crop is None:
            return

        self.cancel_crop_mode()
        self.cropped_image = None
        self.cropped_array = None
        self.pending_crop = None
        self._clear_affine_preview()
        self.save_preview_button.config(state=tk.DISABLED)
        self._update_crop_reset_button_state()
        self.update_preview()

    def apply_crop(self) -> None:
        if self.base_image is None or self.pending_crop is None:
            return

        left, top, right, bottom = self.pending_crop
        if right - left < 2 or bottom - top < 2:
            messagebox.showwarning("トリミング", "トリミング範囲が小さすぎます。")
            return

        self.cropped_image = self.base_image.crop((left, top, right, bottom))
        if self.source_array is not None:
            self.cropped_array = self._crop_source_array(self.source_array, left, top, right, bottom)
        else:
            self.cropped_array = None

        self.pending_crop = None
        self._clear_affine_preview()
        self.save_preview_button.config(state=tk.DISABLED)
        self.cancel_crop_mode()
        self._update_crop_reset_button_state()

    @staticmethod
    def _crop_source_array(array: np.ndarray, left: int, top: int, right: int, bottom: int) -> np.ndarray:
        cropped = array[top:bottom, left:right]
        return np.ascontiguousarray(cropped)

    def _bind_crop_events(self) -> None:
        if self.original_canvas is None:
            return
        self.original_canvas.bind("<ButtonPress-1>", self._on_crop_press)
        self.original_canvas.bind("<B1-Motion>", self._on_crop_drag)
        self.original_canvas.bind("<ButtonRelease-1>", self._on_crop_release)

    def _unbind_crop_events(self) -> None:
        if self.original_canvas is None:
            return
        self.original_canvas.unbind("<ButtonPress-1>")
        self.original_canvas.unbind("<B1-Motion>")
        self.original_canvas.unbind("<ButtonRelease-1>")

    def _on_original_canvas_resize(self, _event: tk.Event) -> None:
        if self.base_image is None:
            return
        self._render_original_canvas(self.base_image)
        self._set_preview_frame_title("original", self.base_image)

    def _on_crop_press(self, event: tk.Event) -> None:
        if not self.crop_mode or self.base_image is None:
            return
        self._crop_drag_start = (event.x, event.y)
        if self.original_canvas is not None:
            self.original_canvas.delete("crop_drag")
            self.original_canvas.delete("crop_rect")

    def _on_crop_drag(self, event: tk.Event) -> None:
        if not self.crop_mode or self._crop_drag_start is None or self.original_canvas is None:
            return
        x0, y0 = self._crop_drag_start
        self.original_canvas.delete("crop_drag")
        self.original_canvas.create_rectangle(
            x0,
            y0,
            event.x,
            event.y,
            outline="#00ff88",
            width=2,
            dash=(4, 4),
            tags="crop_drag",
        )

    def _on_crop_release(self, event: tk.Event) -> None:
        if not self.crop_mode or self._crop_drag_start is None or self.base_image is None:
            return

        if self.original_canvas is not None:
            self.original_canvas.delete("crop_drag")

        start = self._canvas_to_image_point(self._crop_drag_start[0], self._crop_drag_start[1])
        end = self._canvas_to_image_point(event.x, event.y)
        self._crop_drag_start = None
        if start is None or end is None:
            return

        left = max(0, min(start[0], end[0]))
        top = max(0, min(start[1], end[1]))
        right = min(self.base_image.width, max(start[0], end[0]))
        bottom = min(self.base_image.height, max(start[1], end[1]))
        if right - left < 2 or bottom - top < 2:
            self.pending_crop = None
            self.crop_apply_button.config(state=tk.DISABLED)
            return

        self.pending_crop = (left, top, right, bottom)
        self.crop_apply_button.config(state=tk.NORMAL)
        self._draw_crop_overlay()

    def _canvas_to_image_point(self, canvas_x: int, canvas_y: int) -> tuple[int, int] | None:
        if self._original_display_image is None:
            return None

        disp_w, disp_h = self._original_canvas_display_size
        if disp_w <= 0 or disp_h <= 0:
            return None

        local_x = canvas_x - self._original_canvas_offset_x
        local_y = canvas_y - self._original_canvas_offset_y
        local_x = max(0, min(local_x, disp_w))
        local_y = max(0, min(local_y, disp_h))

        image_x = int(round(local_x * self._original_canvas_scale_x))
        image_y = int(round(local_y * self._original_canvas_scale_y))
        image_x = max(0, min(image_x, self._original_display_image.width))
        image_y = max(0, min(image_y, self._original_display_image.height))
        return image_x, image_y

    def _image_to_canvas_point(self, image_x: int, image_y: int) -> tuple[int, int]:
        canvas_x = self._original_canvas_offset_x + image_x / self._original_canvas_scale_x
        canvas_y = self._original_canvas_offset_y + image_y / self._original_canvas_scale_y
        return int(round(canvas_x)), int(round(canvas_y))

    def _draw_crop_overlay(self) -> None:
        if self.original_canvas is None or self.pending_crop is None:
            return

        left, top, right, bottom = self.pending_crop
        x0, y0 = self._image_to_canvas_point(left, top)
        x1, y1 = self._image_to_canvas_point(right, bottom)
        self.original_canvas.delete("crop_rect")
        self.original_canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            outline="#00ff88",
            width=2,
            tags="crop_rect",
        )

    def _render_original_canvas(self, image: Image.Image) -> None:
        if self.original_canvas is None:
            return

        self._original_display_image = image
        self.original_canvas.update_idletasks()
        canvas_w = max(self.original_canvas.winfo_width(), 1)
        canvas_h = max(self.original_canvas.winfo_height(), 1)
        fit_w = min(canvas_w, self.PREVIEW_MAX_SIZE)
        fit_h = min(canvas_h, self.PREVIEW_MAX_SIZE)

        preview = image.copy()
        preview.thumbnail((fit_w, fit_h), Image.Resampling.LANCZOS)
        disp_w, disp_h = preview.size
        if disp_w <= 0 or disp_h <= 0:
            return

        self._original_canvas_scale_x = image.width / disp_w
        self._original_canvas_scale_y = image.height / disp_h
        self._original_canvas_display_size = (disp_w, disp_h)
        self._original_canvas_offset_x = (canvas_w - disp_w) // 2
        self._original_canvas_offset_y = (canvas_h - disp_h) // 2

        self._original_canvas_photo = ImageTk.PhotoImage(preview)
        self.original_canvas.delete("all")
        self.original_canvas.create_image(
            self._original_canvas_offset_x,
            self._original_canvas_offset_y,
            anchor=tk.NW,
            image=self._original_canvas_photo,
            tags="preview_image",
        )
        self._draw_crop_overlay()
