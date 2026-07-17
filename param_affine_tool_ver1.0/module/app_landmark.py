"""特徴点算出（face-alignment / dlib）とプレビュー保存のミックスイン。"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image

from module.face_pipeline import (
    bgr_to_pil,
    draw_dual_landmarks_overlay,
    first_face_landmarks,
    pil_to_bgr,
    warp_face_fs_affine,
)


class LandmarkMixin:
    """特徴点の算出・オーバーレイ/アフィン生成・プレビュー保存を担当する。"""

    def _on_landmark_engine_toggle(self) -> None:
        self._clear_affine_preview()
        if self.pipeline_images.get("processed_overlay") is not None:
            processed = self.pipeline_images.get("processed")
            if processed is not None:
                self._set_preview_image("processed", processed)
            self.pipeline_images["processed_overlay"] = None
            self.save_preview_button.config(state=tk.DISABLED)

    def _preload_landmark_models(self, *, fa_enabled: bool, dlib_enabled: bool) -> None:
        """推論計測にモデル読み込み時間を含めないよう、算出前にモデルを読み込む。"""
        self.landmark_button.config(text="準備中...")
        self.root.update_idletasks()
        if fa_enabled:
            self._get_fa_calculator()
        if dlib_enabled:
            self._get_dlib_calculator()

    def _format_landmark_timing(
        self,
        *,
        fa_enabled: bool,
        dlib_enabled: bool,
        fa_elapsed: float | None,
        dlib_elapsed: float | None,
        yolo_elapsed: float | None,
        total_elapsed: float,
    ) -> str:
        parts: list[str] = []
        if fa_enabled and fa_elapsed is not None:
            parts.append(f"FA {fa_elapsed:.2f}s")
        if dlib_enabled and dlib_elapsed is not None:
            parts.append(f"dlib {dlib_elapsed:.2f}s")
        if yolo_elapsed is not None and yolo_elapsed > 0.0:
            parts.append(f"YOLO {yolo_elapsed:.2f}s")
        if not parts:
            return f"特徴点算出: 合計 {total_elapsed:.2f}s"
        return f"特徴点算出: {', '.join(parts)} (合計 {total_elapsed:.2f}s)"

    def _get_yolo_bbox_scale(self) -> float:
        return self.params["yolo_bbox"].get() / 10.0

    def _update_landmark_timing_label(self) -> None:
        if self.last_landmark_timing_text:
            self.landmark_timing_label.config(text=self.last_landmark_timing_text)
        else:
            self.landmark_timing_label.config(text="")

    def _get_fa_calculator(self):
        if self._fa_calculator is None:
            from module.fa_landmark_calculator import FaceLandmarkCalculator

            self._fa_calculator = FaceLandmarkCalculator(landmark_type="2D", device=self.landmark_device)
        return self._fa_calculator

    def _get_dlib_calculator(self):
        if self._dlib_calculator is None:
            from module.dlib_landmark_calclator import DlibLandmarkCalculator

            self._dlib_calculator = DlibLandmarkCalculator(device=self.landmark_device)
        return self._dlib_calculator

    def run_landmark_analysis(self) -> None:
        if self.base_image is None:
            messagebox.showwarning("特徴点算出", "先に画像を開いてください。")
            return
        if self._landmark_thread is not None and self._landmark_thread.is_alive():
            return

        fa_enabled = self.fa_enabled_var.get()
        dlib_enabled = self.dlib_enabled_var.get()
        if not fa_enabled and not dlib_enabled:
            messagebox.showwarning("特徴点算出", "FA または dlib のいずれかを有効にしてください。")
            return

        self.landmark_button.config(state=tk.DISABLED, text="準備中...")
        try:
            self._preload_landmark_models(fa_enabled=fa_enabled, dlib_enabled=dlib_enabled)
        except Exception as exc:
            self.landmark_button.config(state=tk.NORMAL, text="特徴点算出")
            messagebox.showerror("特徴点算出", f"モデルの読み込みに失敗しました。\n{exc}")
            return

        processed = self._get_processed_image()
        if processed is None:
            messagebox.showwarning("特徴点算出", "処理対象の画像がありません。")
            return
        processed_bgr = pil_to_bgr(processed)
        self.pipeline_images["processed"] = processed
        self._set_preview_image("processed", processed)
        self.landmark_button.config(text="算出中...")
        bbox_scale = self._get_yolo_bbox_scale()

        def worker() -> None:
            errors: list[str] = []
            processed_overlay: Image.Image | None = None
            affine_fa_image: Image.Image | None = None
            affine_dlib_image: Image.Image | None = None
            fa_result: dict | None = None
            dlib_result: dict | None = None
            fa_elapsed: float | None = None
            dlib_elapsed: float | None = None
            yolo_elapsed = 0.0
            started_at = time.perf_counter()

            if fa_enabled:
                try:
                    fa_result, fa_timing = self._get_fa_calculator().predict_from_bgr_timed(
                        processed_bgr, bbox_scale=bbox_scale
                    )
                    fa_elapsed = fa_timing["landmark"]
                    yolo_elapsed += fa_timing["yolo"]
                except Exception as exc:
                    errors.append(f"face-alignment: {exc}")

            if dlib_enabled:
                try:
                    dlib_result, dlib_timing = self._get_dlib_calculator().predict_from_bgr_timed(
                        processed_bgr, bbox_scale=bbox_scale
                    )
                    dlib_elapsed = dlib_timing["landmark"]
                    yolo_elapsed += dlib_timing["yolo"]
                except Exception as exc:
                    errors.append(f"dlib: {exc}")

            if fa_result is not None or dlib_result is not None:
                overlay_bgr = draw_dual_landmarks_overlay(processed_bgr, fa_result, dlib_result)
                processed_overlay = bgr_to_pil(overlay_bgr)

            if fa_enabled:
                fa_landmarks = first_face_landmarks(fa_result) if fa_result is not None else None
                if fa_landmarks is not None:
                    affine_bgr = warp_face_fs_affine(processed_bgr, fa_landmarks)
                    if affine_bgr is not None:
                        affine_fa_image = bgr_to_pil(affine_bgr)

            if dlib_enabled:
                dlib_landmarks = first_face_landmarks(dlib_result) if dlib_result is not None else None
                if dlib_landmarks is not None:
                    affine_bgr = warp_face_fs_affine(processed_bgr, dlib_landmarks)
                    if affine_bgr is not None:
                        affine_dlib_image = bgr_to_pil(affine_bgr)

            total_elapsed = time.perf_counter() - started_at

            self.root.after(
                0,
                lambda: self._on_landmark_analysis_done(
                    fa_enabled=fa_enabled,
                    dlib_enabled=dlib_enabled,
                    fa_elapsed=fa_elapsed,
                    dlib_elapsed=dlib_elapsed,
                    yolo_elapsed=yolo_elapsed,
                    total_elapsed=total_elapsed,
                    processed_overlay=processed_overlay,
                    affine_fa_image=affine_fa_image,
                    affine_dlib_image=affine_dlib_image,
                    errors=errors,
                ),
            )

        self._landmark_thread = threading.Thread(target=worker, daemon=True, name="landmark-analysis")
        self._landmark_thread.start()

    def _on_landmark_analysis_done(
        self,
        *,
        fa_enabled: bool,
        dlib_enabled: bool,
        fa_elapsed: float | None,
        dlib_elapsed: float | None,
        yolo_elapsed: float,
        total_elapsed: float,
        processed_overlay: Image.Image | None,
        affine_fa_image: Image.Image | None,
        affine_dlib_image: Image.Image | None,
        errors: list[str],
    ) -> None:
        self.landmark_button.config(state=tk.NORMAL, text="特徴点算出")
        self._landmark_thread = None
        self.last_landmark_timing_text = self._format_landmark_timing(
            fa_enabled=fa_enabled,
            dlib_enabled=dlib_enabled,
            fa_elapsed=fa_elapsed,
            dlib_elapsed=dlib_elapsed,
            yolo_elapsed=yolo_elapsed,
            total_elapsed=total_elapsed,
        )
        self._update_landmark_timing_label()

        if processed_overlay is not None:
            self.pipeline_images["processed_overlay"] = processed_overlay
            self._set_preview_image("processed", processed_overlay)
        elif fa_enabled or dlib_enabled:
            self.preview_labels["processed"].config(image="", text="特徴点検出失敗")
            processed_ref = self.pipeline_images.get("processed")
            if processed_ref is not None:
                self._set_preview_frame_title("processed", processed_ref)

        if not fa_enabled:
            self.pipeline_images["affine_fa"] = None
            self.preview_labels["affine_fa"].config(image="", text="FA: 無効")
            self._set_preview_frame_title("affine_fa", None)
        elif affine_fa_image is not None:
            self.pipeline_images["affine_fa"] = affine_fa_image
            self._set_preview_image("affine_fa", affine_fa_image)
        else:
            self.pipeline_images["affine_fa"] = None
            self.preview_labels["affine_fa"].config(image="", text="FA: アフィン変換不可")
            self._set_preview_frame_title("affine_fa", None)

        if not dlib_enabled:
            self.pipeline_images["affine_dlib"] = None
            self.preview_labels["affine_dlib"].config(image="", text="dlib: 無効")
            self._set_preview_frame_title("affine_dlib", None)
        elif affine_dlib_image is not None:
            self.pipeline_images["affine_dlib"] = affine_dlib_image
            self._set_preview_image("affine_dlib", affine_dlib_image)
        else:
            self.pipeline_images["affine_dlib"] = None
            self.preview_labels["affine_dlib"].config(image="", text="dlib: アフィン変換不可")
            self._set_preview_frame_title("affine_dlib", None)

        if processed_overlay is not None or affine_fa_image is not None or affine_dlib_image is not None:
            self.save_preview_button.config(state=tk.NORMAL)

        has_success = processed_overlay is not None or affine_fa_image is not None or affine_dlib_image is not None
        if errors and not has_success:
            messagebox.showerror("特徴点算出エラー", "\n".join(errors))
        elif errors:
            messagebox.showwarning("特徴点算出", "一部成功しました。\n" + "\n".join(errors))

    def save_pipeline_previews(self) -> None:
        if self.current_path is None:
            messagebox.showwarning("プレビュー保存", "先に画像を開いてください。")
            return

        default_dir = self.current_path.parent
        out_dir = filedialog.askdirectory(
            title="プレビュー保存先フォルダ",
            initialdir=str(default_dir),
        )
        if not out_dir:
            return

        stem = self.current_path.stem
        processed_plain = self.pipeline_images.get("processed")
        if processed_plain is None and self.base_image is not None:
            processed_plain = self._get_processed_image()

        save_items = [
            (f"{stem}_01_processed.png", processed_plain),
            (f"{stem}_02_processed_landmarks.png", self.pipeline_images.get("processed_overlay")),
            (f"{stem}_03_affine_face_alignment.png", self.pipeline_images.get("affine_fa")),
            (f"{stem}_04_affine_dlib.png", self.pipeline_images.get("affine_dlib")),
        ]

        saved: list[str] = []
        missing: list[str] = []
        for filename, image in save_items:
            if image is None:
                missing.append(filename)
                continue
            path = Path(out_dir) / filename
            image.save(path, format="PNG")
            saved.append(path.name)

        if self.base_image is not None:
            original_path = Path(out_dir) / f"{stem}_00_original.png"
            self.base_image.save(original_path, format="PNG")
            saved.insert(0, original_path.name)

        if self.cropped_image is not None:
            cropped_path = Path(out_dir) / f"{stem}_00_cropped.png"
            self.cropped_image.save(cropped_path, format="PNG")
            if len(saved) >= 1:
                saved.insert(1, cropped_path.name)
            else:
                saved.insert(0, cropped_path.name)

        if not saved:
            messagebox.showwarning("プレビュー保存", "保存できるプレビューがありません。")
            return

        detail = "\n".join(saved)
        if missing:
            detail += "\n\n未生成のためスキップ:\n" + "\n".join(missing)
        messagebox.showinfo("プレビュー保存", f"保存しました ({len(saved)} 件):\n{detail}")
