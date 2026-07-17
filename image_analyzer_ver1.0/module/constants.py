"""アプリ共通の定数・小さなユーティリティ。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

RESIZE_METHODS: dict[str, Image.Resampling] = {
    "NEAREST（最近傍）": Image.Resampling.NEAREST,
    "BILINEAR（双線形）": Image.Resampling.BILINEAR,
    "BICUBIC（双三次）": Image.Resampling.BICUBIC,
    "LANCZOS（高品質）": Image.Resampling.LANCZOS,
    "BOX（ボックス平均）": Image.Resampling.BOX,
    "HAMMING（Hamming窓）": Image.Resampling.HAMMING,
}
DEFAULT_RESIZE_METHOD = "LANCZOS（高品質）"
BLOCK_MEAN_RESIZE_METHOD = "BOX（ボックス平均）"


def default_downloads_dir() -> Path:
    downloads = Path.home() / "Downloads"
    if downloads.is_dir():
        return downloads
    return Path.home()
