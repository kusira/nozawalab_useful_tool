"""アプリ共通の定数（RAWデータ型・リサイズ手法・ファイルダイアログ）。"""

from __future__ import annotations

import numpy as np
from PIL import Image

RAW_DTYPE_OPTIONS = [
    ("uint8", np.uint8),
    ("uint16", np.uint16),
    ("int16", np.int16),
    ("uint32", np.uint32),
    ("int32", np.int32),
    ("float32", np.float32),
    ("float64", np.float64),
]

# 表示名 → PIL Resampling
RESIZE_METHODS: dict[str, Image.Resampling] = {
    "NEAREST（最近傍）": Image.Resampling.NEAREST,
    "BILINEAR（双線形）": Image.Resampling.BILINEAR,
    "BICUBIC（双三次）": Image.Resampling.BICUBIC,
    "LANCZOS（高品質）": Image.Resampling.LANCZOS,
    "BOX（ボックス平均）": Image.Resampling.BOX,
    "HAMMING（Hamming窓）": Image.Resampling.HAMMING,
}
DEFAULT_RESIZE_METHOD = "LANCZOS（高品質）"

FILE_DIALOG_TYPES = [
    ("対応形式", "*.npy *.raw *.png *.jpg *.jpeg"),
    ("NumPy配列", "*.npy"),
    ("RAW画像", "*.raw"),
    ("PNG画像", "*.png"),
    ("JPEG画像", "*.jpg *.jpeg"),
    ("すべてのファイル", "*.*"),
]
