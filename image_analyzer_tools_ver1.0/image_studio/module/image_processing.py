"""表示画像に対する画像処理（見た目の調整）。

ここで扱うのは「表示（プレビュー／画像エクスポート）の見た目」を整える処理で、
統計・ヒストグラム等の解析は生データ（working_array）に対して行う。
そのため ROI 座標がずれないよう、寸法を変えない処理のみを扱う
（リサイズはツールバー、回転・反転は専用ボタンで別途行う）。
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# 互換のため common の手法定義を再輸出する
from common.constants import DEFAULT_RESIZE_METHOD, RESIZE_METHODS  # noqa: F401


def default_params() -> dict[str, int]:
    """画像処理スライダーの初期値。"""
    return {
        "brightness": 100,
        "contrast": 100,
        "gamma": 100,
        "clip_min": 0,
        "clip_max": 100,
        "blur": 0,
        "sharpen": 0,
        "threshold": 0,
        "equalize": 0,
        "invert": 0,
    }


def has_adjustments(params: dict[str, int]) -> bool:
    """初期値から変化があるか（変化が無ければ処理をスキップできる）。"""
    defaults = default_params()
    return any(params.get(k, v) != v for k, v in defaults.items())


def apply_display_adjustments(image: Image.Image, params: dict[str, int]) -> Image.Image:
    """表示用 PIL 画像に、寸法を変えない画像処理を適用する。"""
    result = image.copy()

    clip_min = params.get("clip_min", 0) / 100.0
    clip_max = params.get("clip_max", 100) / 100.0
    if clip_min > clip_max:
        clip_min, clip_max = clip_max, clip_min
    if clip_min > 0.0 or clip_max < 1.0:
        arr = np.asarray(result, dtype=np.float32) / 255.0
        low = clip_min
        high = max(clip_max, low + 1e-6)
        arr = np.clip((arr - low) / (high - low), 0.0, 1.0)
        result = Image.fromarray((arr * 255).astype(np.uint8), mode=result.mode)

    brightness = params.get("brightness", 100) / 100.0
    contrast = params.get("contrast", 100) / 100.0
    if abs(brightness - 1.0) > 1e-3:
        result = ImageEnhance.Brightness(result).enhance(brightness)
    if abs(contrast - 1.0) > 1e-3:
        result = ImageEnhance.Contrast(result).enhance(contrast)

    gamma = max(params.get("gamma", 100) / 100.0, 0.01)
    if abs(gamma - 1.0) > 1e-3:
        arr = np.asarray(result, dtype=np.float32) / 255.0
        arr = np.power(arr, gamma)
        result = Image.fromarray((arr * 255).astype(np.uint8), mode=result.mode)

    equalize_strength = params.get("equalize", 0) / 100.0
    if equalize_strength > 0.0:
        equalized = ImageOps.equalize(result.convert("L")).convert(result.mode)
        result = Image.blend(result, equalized, equalize_strength)

    blur_radius = params.get("blur", 0)
    if blur_radius > 0:
        result = result.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    sharpen_amount = params.get("sharpen", 0) / 100.0
    if sharpen_amount > 0.0:
        sharpened = result.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        result = Image.blend(result, sharpened, sharpen_amount)

    threshold = params.get("threshold", 0)
    if threshold > 0:
        gray = result.convert("L")
        binary = gray.point(lambda p: 255 if p >= threshold else 0, mode="L")
        result = binary.convert(result.mode)

    invert_strength = params.get("invert", 0) / 100.0
    if invert_strength > 0.0:
        inverted = ImageOps.invert(result.convert("RGB"))
        if result.mode != "RGB":
            inverted = inverted.convert(result.mode)
        result = Image.blend(result, inverted, invert_strength)

    return result
