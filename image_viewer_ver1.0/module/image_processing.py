"""画像処理パラメータの適用。"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def default_params() -> dict[str, int]:
    return {
        "resize": 10,
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


def apply_processing(image: Image.Image, params: dict[str, int]) -> Image.Image:
    resize = params.get("resize", 10) / 10.0
    if resize < 0.999:
        new_w = max(1, int(round(image.width * resize)))
        new_h = max(1, int(round(image.height * resize)))
        result = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    else:
        result = image.copy()

    clip_min = params["clip_min"] / 100.0
    clip_max = params["clip_max"] / 100.0
    if clip_min > clip_max:
        clip_min, clip_max = clip_max, clip_min

    if clip_min > 0.0 or clip_max < 1.0:
        arr = np.asarray(result, dtype=np.float32) / 255.0
        low = clip_min
        high = max(clip_max, low + 1e-6)
        arr = np.clip((arr - low) / (high - low), 0.0, 1.0)
        result = Image.fromarray((arr * 255).astype(np.uint8), mode=result.mode)

    brightness = params["brightness"] / 100.0
    contrast = params["contrast"] / 100.0
    result = ImageEnhance.Brightness(result).enhance(brightness)
    result = ImageEnhance.Contrast(result).enhance(contrast)

    gamma = max(params["gamma"] / 100.0, 0.01)
    if abs(gamma - 1.0) > 1e-3:
        arr = np.asarray(result, dtype=np.float32) / 255.0
        arr = np.power(arr, gamma)
        result = Image.fromarray((arr * 255).astype(np.uint8), mode=result.mode)

    equalize_strength = params["equalize"] / 100.0
    if equalize_strength > 0.0:
        equalized = ImageOps.equalize(result.convert("L")).convert(result.mode)
        result = Image.blend(result, equalized, equalize_strength)

    blur_radius = params["blur"]
    if blur_radius > 0:
        result = result.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    sharpen_amount = params["sharpen"] / 100.0
    if sharpen_amount > 0.0:
        sharpened = result.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        result = Image.blend(result, sharpened, sharpen_amount)

    threshold = params["threshold"]
    if threshold > 0:
        gray = result.convert("L")
        binary = gray.point(lambda p: 255 if p >= threshold else 0, mode="L")
        result = binary.convert(result.mode)

    invert_strength = params["invert"] / 100.0
    if invert_strength > 0.0:
        inverted = ImageOps.invert(result.convert("RGB"))
        if result.mode != "RGB":
            inverted = inverted.convert(result.mode)
        result = Image.blend(result, inverted, invert_strength)

    rotate_angle = params["rotate"]
    if rotate_angle != 0:
        result = result.rotate(rotate_angle, expand=True, fillcolor=0)

    return result
