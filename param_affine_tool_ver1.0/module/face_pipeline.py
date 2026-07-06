"""顔ランドマーク描画とアフィン変換のユーティリティ。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from module.fs_affine_convert import default_mean_front_parts
AFFINE_SIZE_WIDTH = 250
AFFINE_SIZE_HEIGHT = 250
DEFAULT_AFFINE_OUTPUT_SIZE = (AFFINE_SIZE_WIDTH, AFFINE_SIZE_HEIGHT)
MAX_AFFINE_SIZE = max(AFFINE_SIZE_WIDTH, AFFINE_SIZE_HEIGHT)
DEFAULT_AF_SIZE = (MAX_AFFINE_SIZE, MAX_AFFINE_SIZE)
DEFAULT_AFFINE_SQUARE = True

_affine_converter: object | None = None
_affine_converter_key: tuple[object, ...] | None = None


def expand_bbox_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    scale: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """YOLO bbox を中心基準で scale 倍のサイズに拡大する。"""
    if scale <= 1.0:
        return x1, y1, x2, y2
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_w = (x2 - x1) * 0.5 * scale
    half_h = (y2 - y1) * 0.5 * scale
    return (
        max(0.0, cx - half_w),
        max(0.0, cy - half_h),
        min(float(width - 1), cx + half_w),
        min(float(height - 1), cy + half_h),
    )


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _apply_landmarks_inplace(
    out: np.ndarray,
    prediction: dict[str, Any],
    color: tuple[int, int, int],
    *,
    label: str | None = None,
) -> None:
    faces = prediction.get("faces", [])
    if not faces:
        if label:
            cv2.putText(
                out,
                f"{label}: not detected",
                (20, 40 if label == "FA" else 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
        return

    face = faces[0]
    pts = np.array(face["landmarks"], dtype=np.int32)
    for x, y in pts:
        cv2.circle(out, (int(x), int(y)), 2, color, -1, lineType=cv2.LINE_AA)
    for a, b in face.get("edges", []):
        xa, ya = pts[a]
        xb, yb = pts[b]
        cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), color, 1, lineType=cv2.LINE_AA)

    xs = pts[:, 0]
    ys = pts[:, 1]
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
    if label:
        cv2.putText(
            out,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_landmarks_overlay(
    img_bgr: np.ndarray,
    prediction: dict[str, Any],
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """ランドマーク推定結果を画像にオーバーレイする。"""
    out = img_bgr.copy()
    _apply_landmarks_inplace(out, prediction, color)
    if not prediction.get("faces"):
        cv2.putText(
            out,
            "face not detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def draw_dual_landmarks_overlay(
    img_bgr: np.ndarray,
    fa_prediction: dict[str, Any] | None,
    dlib_prediction: dict[str, Any] | None,
) -> np.ndarray:
    """画像処理後画像に face-alignment と dlib の特徴点を重ねて描画する。"""
    out = img_bgr.copy()
    if fa_prediction is not None:
        _apply_landmarks_inplace(out, fa_prediction, (0, 255, 0), label="FA")
    if dlib_prediction is not None:
        _apply_landmarks_inplace(out, dlib_prediction, (255, 128, 0), label="dlib")
    return out


def first_face_landmarks(prediction: dict[str, Any]) -> list[list[float]] | None:
    faces = prediction.get("faces", [])
    if not faces:
        return None
    return faces[0].get("landmarks")


def get_fs_affine_converter(
    afsize: tuple[int, int] | None = None,
    square: bool = DEFAULT_AFFINE_SQUARE,
    temp_parts: np.ndarray | None = None,
):
    """affine_convert インスタンスを返す（設定が同じなら再利用）。"""
    global _affine_converter, _affine_converter_key

    from module.fs_affine_convert import affine_convert

    if afsize is None:
        afsize = DEFAULT_AF_SIZE

    template_parts = temp_parts if temp_parts is not None else default_mean_front_parts()

    key = (tuple(afsize), square, template_parts.tobytes())
    if _affine_converter is None or _affine_converter_key != key:
        _affine_converter = affine_convert(list(afsize), square, template_parts)
        _affine_converter_key = key
    return _affine_converter


def warp_face_fs_affine(
    img_bgr: np.ndarray,
    landmarks: list[list[float]],
    *,
    affine_size_width: int = AFFINE_SIZE_WIDTH,
    affine_size_height: int = AFFINE_SIZE_HEIGHT,
    square: bool = DEFAULT_AFFINE_SQUARE,
    binary_on: bool = False,
    temp_parts: np.ndarray | None = None,
) -> np.ndarray | None:
    """landmark-tool と同様の fs_affine_convert パイプラインでアフィン変換する。"""
    if len(landmarks) < 68:
        return None

    max_affine_size = max(affine_size_width, affine_size_height)
    afsize = (max_affine_size, max_affine_size)
    converter = get_fs_affine_converter(
        afsize=afsize,
        square=square,
        temp_parts=temp_parts,
    )
    landmark = np.round(np.asarray(landmarks[:68], dtype=np.float64))

    src = img_bgr
    if src.ndim == 2:
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    affine_image = converter.main(src, landmark, binary_on=binary_on)
    if affine_image is None:
        return None

    affine_image = np.array(affine_image, dtype=np.uint8)
    return cv2.resize(affine_image, (affine_size_width, affine_size_height))


def warp_face_affine(
    img_bgr: np.ndarray,
    landmarks: list[list[float]],
    *,
    affine_size_width: int = AFFINE_SIZE_WIDTH,
    affine_size_height: int = AFFINE_SIZE_HEIGHT,
) -> np.ndarray | None:
    """後方互換のエイリアス。fs_affine_convert を使用する。"""
    return warp_face_fs_affine(
        img_bgr,
        landmarks,
        affine_size_width=affine_size_width,
        affine_size_height=affine_size_height,
        square=DEFAULT_AFFINE_SQUARE,
    )
