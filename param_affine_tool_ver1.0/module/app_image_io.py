"""画像ファイル（NPY/RAW/PNG/JPEG）の読み込みと配列変換・メタ情報のミックスイン。"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image

from module.constants import RAW_DTYPE_OPTIONS


class ImageIOMixin:
    """各種画像形式の読み込み・配列/画像変換・メタ情報表示を担当する。"""

    @staticmethod
    def load_npy(path: Path) -> np.ndarray:
        return np.load(path)

    @staticmethod
    def load_raster(path: Path) -> tuple[np.ndarray, Image.Image]:
        with Image.open(path) as img:
            image = img.convert("RGB" if img.mode not in ("L", "LA") else "L")
        return np.asarray(image), image

    load_png = load_raster

    @staticmethod
    def _resolve_raw_dtype(dtype_name: str, endian: str) -> np.dtype:
        base_dtype = next(dt for name, dt in RAW_DTYPE_OPTIONS if name == dtype_name)
        np_dtype = np.dtype(base_dtype)
        if np_dtype.itemsize > 1:
            byteorder = "<" if endian == "little" else ">"
            np_dtype = np_dtype.newbyteorder(byteorder)
        return np_dtype

    @staticmethod
    def load_raw(path: Path, settings: dict[str, object]) -> np.ndarray:
        width = int(settings["width"])
        height = int(settings["height"])
        channels = int(settings["channels"])
        offset = int(settings["offset"])
        dtype_name = str(settings["dtype"])
        endian = str(settings["endian"])

        dtype = ImageIOMixin._resolve_raw_dtype(dtype_name, endian)

        file_size = path.stat().st_size
        if offset >= file_size:
            raise ValueError("オフセットがファイルサイズ以上です。")

        count = width * height * channels
        available_bytes = file_size - offset
        required_bytes = count * dtype.itemsize
        if available_bytes < required_bytes:
            raise ValueError(
                f"データサイズが不足しています。必要: {required_bytes:,} bytes, 利用可能: {available_bytes:,} bytes"
            )

        data = np.fromfile(path, dtype=dtype, count=count, offset=offset)
        if channels == 1:
            return data.reshape(height, width)
        return data.reshape(height, width, channels)

    @staticmethod
    def _guess_raw_settings(path: Path, current: dict[str, object]) -> dict[str, object]:
        file_size = path.stat().st_size
        offset = int(current.get("offset", 0))
        dtype_name = str(current.get("dtype", "uint16"))
        dtype = np.dtype(next(dt for name, dt in RAW_DTYPE_OPTIONS if name == dtype_name))
        channels = max(1, int(current.get("channels", 1)))
        available = max(file_size - offset, 0)
        element_count = available // dtype.itemsize if available else 0

        width = int(current.get("width", 1600))
        height = int(current.get("height", 1300))
        if width > 0 and height > 0 and width * height * channels == element_count:
            return {
                "width": width,
                "height": height,
                "channels": channels,
                "offset": offset,
                "dtype": dtype_name,
                "endian": current.get("endian", "little"),
            }

        common_sizes = [256, 512, 640, 720, 768, 1024, 1280, 1920, 2048]
        for side in common_sizes:
            if side * side * channels * dtype.itemsize + offset == file_size:
                return {
                    "width": side,
                    "height": side,
                    "channels": channels,
                    "offset": offset,
                    "dtype": dtype_name,
                    "endian": current.get("endian", "little"),
                }

        if element_count > 0:
            side = int(np.sqrt(element_count // channels))
            if side > 0 and side * side * channels == element_count:
                width = side
                height = side

        return {
            "width": width,
            "height": height,
            "channels": channels,
            "offset": offset,
            "dtype": dtype_name,
            "endian": current.get("endian", "little"),
        }

    @staticmethod
    def array_to_image(array: np.ndarray) -> Image.Image:
        arr = np.asarray(array)

        if arr.ndim == 1:
            side = int(np.sqrt(arr.size))
            if side * side != arr.size:
                raise ValueError("1次元配列は正方形サイズである必要があります。")
            arr = arr.reshape(side, side)

        if arr.ndim > 3:
            arr = np.squeeze(arr)
            if arr.ndim > 3:
                raise ValueError(f"対応していない配列次元です: {array.shape}")

        if arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
                arr = np.moveaxis(arr, 0, -1)
            channels = arr.shape[-1]
            if channels == 1:
                arr = arr[..., 0]
            elif channels >= 3:
                arr = arr[..., :3]

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = arr.astype(np.float64)
        arr -= arr.min()
        max_val = arr.max()
        if max_val > 0:
            arr /= max_val
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        return Image.fromarray(arr, mode="RGB")

    def _update_meta(self, array: np.ndarray, path: Path | None = None) -> None:
        lines = []
        if path is not None:
            lines.append(f"file: {path.name}")
            if self.current_file_type == "raw":
                lines.extend(
                    [
                        f"raw width: {self.raw_settings['width']}",
                        f"raw height: {self.raw_settings['height']}",
                        f"raw channels: {self.raw_settings['channels']}",
                        f"raw offset: {self.raw_settings['offset']}",
                        f"raw dtype: {self.raw_settings['dtype']}",
                        f"raw endian: {self.raw_settings['endian']}",
                    ]
                )
        lines.extend(
            [
                f"shape: {array.shape}",
                f"dtype: {array.dtype}",
                f"min: {np.nanmin(array):.6g}",
                f"max: {np.nanmax(array):.6g}",
                f"mean: {np.nanmean(array):.6g}",
                f"std: {np.nanstd(array):.6g}",
            ]
        )
        if array.ndim >= 3:
            lines.append(f"channels: {array.shape[-1]}")

        self.meta_text.config(state=tk.NORMAL)
        self.meta_text.delete("1.0", tk.END)
        self.meta_text.insert(tk.END, "\n".join(lines))
        self.meta_text.config(state=tk.DISABLED)
