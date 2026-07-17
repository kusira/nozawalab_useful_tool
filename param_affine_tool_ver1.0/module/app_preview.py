"""プレビュー更新・画像処理適用・パラメータリセットのミックスイン。"""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageTk

from module.constants import DEFAULT_RESIZE_METHOD, RESIZE_METHODS
from module.face_pipeline import AFFINE_SIZE_HEIGHT, AFFINE_SIZE_WIDTH


class PreviewMixin:
    """4分割プレビューの描画・画像処理適用・スライダーのリセットを担当する。"""

    def reset_params(self, update_preview: bool = True) -> None:
        defaults = {
            "resize": 10,
            "yolo_bbox": 10,
            "brightness": 100,
            "contrast": 100,
            "gamma": 100,
            "blur": 0,
            "sharpen": 0,
            "threshold": 0,
            "rotate": 0,
            "clip_min": 0,
            "clip_max": 100,
            "equalize": 0,
            "invert": 0,
        }
        for key, value in defaults.items():
            self._set_param(key, value)
        self.resize_method_var.set(DEFAULT_RESIZE_METHOD)

        if update_preview:
            self.update_preview()

    def reset_all(self) -> None:
        """スライダー・トリミング・特徴点結果をまとめて初期化する。"""
        self.cancel_crop_mode()
        self.cropped_image = None
        self.cropped_array = None
        self.pending_crop = None
        self._update_crop_reset_button_state()
        self.reset_params(update_preview=False)
        self._clear_affine_preview()
        self.save_preview_button.config(state=tk.DISABLED)
        self.last_landmark_timing_text = None
        self._update_landmark_timing_label()
        if self.base_image is not None:
            self.update_preview()

    def schedule_update(self) -> None:
        if self._update_job is not None:
            self.root.after_cancel(self._update_job)
        self._update_job = self.root.after(80, self._run_scheduled_update)

    def _run_scheduled_update(self) -> None:
        self._update_job = None
        self.update_preview()

    def update_preview(self) -> None:
        if self.base_image is None:
            return

        self._render_original_canvas(self.base_image)
        self._set_preview_frame_title("original", self.base_image)

        processed = self._get_processed_image()
        if processed is None:
            self._set_processed_placeholder("画像を開いてください")
            return

        self.pipeline_images["processed"] = processed
        self.pipeline_images["processed_overlay"] = None
        self._set_preview_image("processed", processed)
        if not self.crop_mode:
            self._clear_affine_preview()

    def _set_processed_placeholder(self, text: str) -> None:
        self.preview_photos["processed"] = None
        self.preview_labels["processed"].config(image="", text=text)
        self._set_preview_frame_title("processed", None)

    def _get_resize_scale(self) -> float:
        return self.params["resize"].get() / 10.0

    def _get_resize_resampling(self) -> Image.Resampling:
        name = self.resize_method_var.get()
        return RESIZE_METHODS.get(name, RESIZE_METHODS[DEFAULT_RESIZE_METHOD])

    def _scale_image(self, image: Image.Image) -> Image.Image:
        scale = self._get_resize_scale()
        if scale >= 0.999:
            return image.copy()
        new_w = max(1, int(round(image.width * scale)))
        new_h = max(1, int(round(image.height * scale)))
        if new_w == image.width and new_h == image.height:
            return image.copy()
        return image.resize((new_w, new_h), self._get_resize_resampling())

    def _get_working_image(self) -> Image.Image | None:
        """トリミング適用済みならその画像、未適用なら元画像を返す。"""
        if self.cropped_image is not None:
            return self.cropped_image
        return self.base_image

    def _get_resized_crop(self) -> Image.Image | None:
        working = self._get_working_image()
        if working is None:
            return None
        return self._scale_image(working.copy())

    def _get_processed_image(self) -> Image.Image | None:
        resized = self._get_resized_crop()
        if resized is None:
            return None
        return self.apply_processing(resized)

    @staticmethod
    def _format_image_resolution(image: Image.Image) -> str:
        return f"{image.width} × {image.height}"

    def _preview_base_title(self, key: str) -> str:
        for preview_key, title in self.PREVIEW_SPECS:
            if preview_key == key:
                return title
        return key

    def _affine_preview_title(self, key: str, image: Image.Image | None = None) -> str:
        base = self._preview_base_title(key)
        if image is not None:
            return f"{base} ({self._format_image_resolution(image)})"
        return f"{base} ({AFFINE_SIZE_WIDTH} × {AFFINE_SIZE_HEIGHT})"

    def _set_preview_frame_title(self, key: str, image: Image.Image | None) -> None:
        frame = self.preview_frames.get(key)
        if frame is None:
            return
        if key in ("original", "processed") and image is not None:
            frame.config(text=f"{self._preview_base_title(key)} ({self._format_image_resolution(image)})")
        elif key in ("affine_fa", "affine_dlib"):
            frame.config(text=self._affine_preview_title(key, image))
        else:
            frame.config(text=self._preview_base_title(key))

    def _set_preview_image(self, key: str, image: Image.Image) -> None:
        if key == "original":
            self._render_original_canvas(image)
            self._set_preview_frame_title(key, image)
            return

        photo = self.to_photo(image)
        self.preview_photos[key] = photo
        label = self.preview_labels[key]
        label.config(image=photo, text="")
        if key in ("original", "processed", "affine_fa", "affine_dlib"):
            self._set_preview_frame_title(key, image)

    def _affine_preview_placeholder(self, engine: str) -> str:
        if engine == "fa":
            return "FA: 無効" if not self.fa_enabled_var.get() else "「特徴点算出」で生成"
        return "dlib: 無効" if not self.dlib_enabled_var.get() else "「特徴点算出」で生成"

    def _clear_affine_preview(self) -> None:
        for key, engine in (("affine_fa", "fa"), ("affine_dlib", "dlib")):
            self.pipeline_images[key] = None
            self.preview_photos[key] = None
            self.preview_labels[key].config(image="", text=self._affine_preview_placeholder(engine))
            self._set_preview_frame_title(key, None)

    def apply_processing(self, image: Image.Image) -> Image.Image:
        result = image.copy()

        clip_min = self.params["clip_min"].get() / 100.0
        clip_max = self.params["clip_max"].get() / 100.0
        if clip_min > clip_max:
            clip_min, clip_max = clip_max, clip_min

        if clip_min > 0.0 or clip_max < 1.0:
            arr = np.asarray(result, dtype=np.float32) / 255.0
            low = clip_min
            high = max(clip_max, low + 1e-6)
            arr = np.clip((arr - low) / (high - low), 0.0, 1.0)
            result = Image.fromarray((arr * 255).astype(np.uint8), mode=result.mode)

        brightness = self.params["brightness"].get() / 100.0
        contrast = self.params["contrast"].get() / 100.0
        result = ImageEnhance.Brightness(result).enhance(brightness)
        result = ImageEnhance.Contrast(result).enhance(contrast)

        gamma = max(self.params["gamma"].get() / 100.0, 0.01)
        if abs(gamma - 1.0) > 1e-3:
            arr = np.asarray(result, dtype=np.float32) / 255.0
            arr = np.power(arr, gamma)
            result = Image.fromarray((arr * 255).astype(np.uint8), mode=result.mode)

        equalize_strength = self.params["equalize"].get() / 100.0
        if equalize_strength > 0.0:
            equalized = ImageOps.equalize(result.convert("L")).convert(result.mode)
            result = Image.blend(result, equalized, equalize_strength)

        blur_radius = self.params["blur"].get()
        if blur_radius > 0:
            result = result.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        sharpen_amount = self.params["sharpen"].get() / 100.0
        if sharpen_amount > 0.0:
            sharpened = result.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            result = Image.blend(result, sharpened, sharpen_amount)

        threshold = self.params["threshold"].get()
        if threshold > 0:
            gray = result.convert("L")
            binary = gray.point(lambda p: 255 if p >= threshold else 0, mode="L")
            result = binary.convert(result.mode)

        invert_strength = self.params["invert"].get() / 100.0
        if invert_strength > 0.0:
            inverted = ImageOps.invert(result.convert("RGB"))
            if result.mode != "RGB":
                inverted = inverted.convert(result.mode)
            result = Image.blend(result, inverted, invert_strength)

        rotate_angle = self.params["rotate"].get()
        if rotate_angle != 0:
            result = result.rotate(rotate_angle, expand=True, fillcolor=0)

        return result

    def to_photo(self, image: Image.Image) -> ImageTk.PhotoImage:
        preview = image.copy()
        preview.thumbnail((self.PREVIEW_MAX_SIZE, self.PREVIEW_MAX_SIZE), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(preview)
