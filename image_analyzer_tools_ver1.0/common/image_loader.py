"""NPY / RAW / PNG / JPEG などの読み込みユーティリティ。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from common.constants import RAW_DTYPE_OPTIONS

SUPPORTED_EXTENSIONS = {".npy", ".raw", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_RAW_SETTINGS: dict[str, object] = {
    "width": 1600,
    "height": 1300,
    "channels": 1,
    "offset": 0,
    "dtype": "uint16",
    "endian": "little",
}


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def collect_image_paths(target: Path) -> list[Path]:
    return collect_image_paths_recursive(target)


def collect_image_paths_recursive(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if is_supported_image(target) else []
    if not target.is_dir():
        return []
    paths: list[Path] = []
    try:
        for item in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if item.is_file() and is_supported_image(item):
                paths.append(item)
            elif item.is_dir():
                paths.extend(collect_image_paths_recursive(item))
    except OSError:
        return paths
    return paths


def load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def load_raster(path: Path) -> tuple[np.ndarray, Image.Image]:
    with Image.open(path) as img:
        image = img.convert("RGB" if img.mode not in ("L", "LA") else "L")
    return np.asarray(image), image


def resolve_raw_dtype(dtype_name: str, endian: str) -> np.dtype:
    base_dtype = next(dt for name, dt in RAW_DTYPE_OPTIONS if name == dtype_name)
    np_dtype = np.dtype(base_dtype)
    if np_dtype.itemsize > 1:
        byteorder = "<" if endian == "little" else ">"
        np_dtype = np_dtype.newbyteorder(byteorder)
    return np_dtype


def load_raw(path: Path, settings: dict[str, object]) -> np.ndarray:
    width = int(settings["width"])
    height = int(settings["height"])
    channels = int(settings["channels"])
    offset = int(settings["offset"])
    dtype_name = str(settings["dtype"])
    endian = str(settings["endian"])

    dtype = resolve_raw_dtype(dtype_name, endian)

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


def guess_raw_settings(path: Path, current: dict[str, object] | None = None) -> dict[str, object]:
    current = current or DEFAULT_RAW_SETTINGS
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


def load_image_file(
    path: Path,
    raw_settings: dict[str, object] | None = None,
) -> tuple[np.ndarray, Image.Image, str]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = load_npy(path)
        return array, array_to_image(array), "npy"
    if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
        array, image = load_raster(path)
        return array, image, "raster"
    settings = raw_settings or guess_raw_settings(path, DEFAULT_RAW_SETTINGS)
    array = load_raw(path, settings)
    return array, array_to_image(array), "raw"


def array_stats(array: np.ndarray) -> dict[str, object]:
    flat = array.astype(np.float64).ravel()
    return {
        "shape": array.shape,
        "dtype": str(array.dtype),
        "min": float(flat.min()) if flat.size else 0.0,
        "max": float(flat.max()) if flat.size else 0.0,
        "mean": float(flat.mean()) if flat.size else 0.0,
        "std": float(flat.std()) if flat.size else 0.0,
    }
