"""配列のリサイズ（ブロック平均 / PIL 補間）。"""

from __future__ import annotations

import numpy as np
from PIL import Image

from common.constants import BLOCK_MEAN_RESIZE_METHOD, RESIZE_METHODS


def _block_mean_2d(
    src: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
) -> np.ndarray:
    """各出力画素に対応する矩形ブロックの平均値を積分画像で計算する。"""
    h, w = src.shape
    padded = np.zeros((h + 1, w + 1), dtype=np.float64)
    padded[1:, 1:] = src
    integral = padded.cumsum(0).cumsum(1)

    iy0 = y0[:, None]
    iy1 = y1[:, None]
    ix0 = x0[None, :]
    ix1 = x1[None, :]
    sums = integral[iy1, ix1] - integral[iy0, ix1] - integral[iy1, ix0] + integral[iy0, ix0]
    area = (iy1 - iy0).astype(np.float64) * (ix1 - ix0).astype(np.float64)
    return sums / np.maximum(area, 1.0)


def resize_array_block_mean(array: np.ndarray, scale: float) -> np.ndarray:
    """配列を scale 倍に縮小（対応ブロックの周辺平均）。"""
    arr = np.asarray(array)
    if scale >= 0.999:
        return arr
    h, w = arr.shape[:2]
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    if nh == h and nw == w:
        return arr

    y0 = (np.arange(nh) * h) // nh
    y1 = ((np.arange(nh) + 1) * h) // nh
    x0 = (np.arange(nw) * w) // nw
    x1 = ((np.arange(nw) + 1) * w) // nw
    y1 = np.maximum(y1, y0 + 1)
    x1 = np.maximum(x1, x0 + 1)

    src = arr.astype(np.float64, copy=False)
    if arr.ndim == 2:
        out = _block_mean_2d(src, y0, y1, x0, x1)
    else:
        channels = [_block_mean_2d(src[..., c], y0, y1, x0, x1) for c in range(arr.shape[-1])]
        out = np.stack(channels, axis=-1)

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        out = np.clip(np.rint(out), info.min, info.max).astype(arr.dtype)
    else:
        out = out.astype(arr.dtype, copy=False)
    return out


def resize_array_pil(array: np.ndarray, scale: float, resampling: Image.Resampling) -> np.ndarray:
    arr = np.asarray(array)
    if scale >= 0.999:
        return arr
    h, w = arr.shape[:2]
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    if nh == h and nw == w:
        return arr

    original_dtype = arr.dtype
    is_int = np.issubdtype(original_dtype, np.integer)

    def resize_plane(plane: np.ndarray) -> np.ndarray:
        # PIL の "F" モードは float32。float64 を渡すとバッファが誤読されノイズ化する。
        pil = Image.fromarray(np.ascontiguousarray(plane, dtype=np.float32), mode="F")
        return np.array(pil.resize((nw, nh), resampling), dtype=np.float64)

    if arr.ndim == 2:
        out = resize_plane(arr)
    else:
        out = np.stack([resize_plane(arr[..., c]) for c in range(arr.shape[-1])], axis=-1)

    if is_int:
        info = np.iinfo(original_dtype)
        out = np.clip(np.rint(out), info.min, info.max).astype(original_dtype)
    else:
        out = out.astype(original_dtype, copy=False)
    return out


def resize_array(array: np.ndarray, scale: float, method_name: str) -> np.ndarray:
    """手法名に応じて配列をリサイズする。"""
    if method_name == BLOCK_MEAN_RESIZE_METHOD:
        return resize_array_block_mean(array, scale)
    resampling = RESIZE_METHODS.get(method_name, RESIZE_METHODS[BLOCK_MEAN_RESIZE_METHOD])
    return resize_array_pil(array, scale, resampling)
