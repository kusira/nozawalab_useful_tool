"""解析結果の可視化（カラーマップ・ヒートマップ・プロット画像）。"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from module.analysis import COLORMAPS, to_float_gray


def _lerp_palette(stops: list[tuple[float, tuple[int, int, int]]], n: int = 256) -> np.ndarray:
    """色停留点から LUT を生成する。"""
    lut = np.zeros((n, 3), dtype=np.uint8)
    positions = [s[0] for s in stops]
    colors = np.array([s[1] for s in stops], dtype=np.float64)
    xs = np.linspace(0.0, 1.0, n)
    for c in range(3):
        lut[:, c] = np.clip(np.interp(xs, positions, colors[:, c]), 0, 255).astype(np.uint8)
    return lut


_LUTS: dict[str, np.ndarray] = {
    "gray": _lerp_palette([(0, (0, 0, 0)), (1, (255, 255, 255))]),
    "viridis": _lerp_palette(
        [
            (0.0, (68, 1, 84)),
            (0.25, (59, 82, 139)),
            (0.5, (33, 145, 140)),
            (0.75, (94, 201, 98)),
            (1.0, (253, 231, 37)),
        ]
    ),
    "plasma": _lerp_palette(
        [
            (0.0, (13, 8, 135)),
            (0.25, (126, 3, 168)),
            (0.5, (204, 71, 120)),
            (0.75, (248, 149, 64)),
            (1.0, (240, 249, 33)),
        ]
    ),
    "inferno": _lerp_palette(
        [
            (0.0, (0, 0, 4)),
            (0.25, (87, 16, 110)),
            (0.5, (188, 55, 84)),
            (0.75, (249, 142, 9)),
            (1.0, (252, 255, 164)),
        ]
    ),
    "magma": _lerp_palette(
        [
            (0.0, (0, 0, 4)),
            (0.25, (81, 18, 124)),
            (0.5, (183, 55, 121)),
            (0.75, (252, 137, 97)),
            (1.0, (252, 253, 191)),
        ]
    ),
    "jet": _lerp_palette(
        [
            (0.0, (0, 0, 128)),
            (0.125, (0, 0, 255)),
            (0.375, (0, 255, 255)),
            (0.625, (255, 255, 0)),
            (0.875, (255, 0, 0)),
            (1.0, (128, 0, 0)),
        ]
    ),
    "hot": _lerp_palette(
        [
            (0.0, (0, 0, 0)),
            (0.33, (255, 0, 0)),
            (0.66, (255, 255, 0)),
            (1.0, (255, 255, 255)),
        ]
    ),
    "cool": _lerp_palette([(0.0, (0, 255, 255)), (1.0, (255, 0, 255))]),
    "turbo": _lerp_palette(
        [
            (0.0, (48, 18, 59)),
            (0.2, (70, 130, 180)),
            (0.4, (53, 183, 121)),
            (0.6, (210, 210, 50)),
            (0.8, (245, 125, 40)),
            (1.0, (122, 4, 3)),
        ]
    ),
}


def apply_colormap(array: np.ndarray, cmap: str = "viridis") -> Image.Image:
    """単チャンネル配列を疑似カラー画像にする。"""
    if cmap not in _LUTS:
        cmap = "viridis"
    gray = to_float_gray(array)
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return Image.new("RGB", (gray.shape[1], gray.shape[0]), (0, 0, 0))
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmax <= vmin:
        norm = np.zeros_like(gray, dtype=np.float64)
    else:
        norm = (gray - vmin) / (vmax - vmin)
    idx = np.clip((norm * 255).astype(int), 0, 255)
    lut = _LUTS[cmap]
    rgb = lut[idx]
    return Image.fromarray(rgb, mode="RGB")


def abs_diff_heatmap(abs_diff: np.ndarray, cmap: str = "hot") -> Image.Image:
    return apply_colormap(abs_diff, cmap=cmap)


def array_to_display_image(array: np.ndarray, cmap: str | None = None) -> Image.Image:
    """表示用画像。cmap 指定時は疑似カラー、未指定は通常正規化。"""
    if cmap and cmap != "gray" and cmap in COLORMAPS:
        return apply_colormap(array, cmap=cmap)

    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        rgb = arr[..., :3].astype(np.float64)
        for c in range(3):
            ch = rgb[..., c]
            mn, mx = float(np.nanmin(ch)), float(np.nanmax(ch))
            if mx > mn:
                rgb[..., c] = (ch - mn) / (mx - mn) * 255.0
            else:
                rgb[..., c] = 0
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")

    return apply_colormap(arr, cmap="gray")


def _fmt_axis_num(v: float) -> str:
    """軸ラベル用の数値整形。画素値・カウントは整数なので小数は使わない。"""
    v = round(float(v))
    if v == 0:
        return "0"
    if abs(v) >= 100000:
        return f"{v:.1e}"
    return f"{int(v)}"


def render_histogram_image(
    hist: dict[str, Any],
    width: int = 460,
    height: int = 260,
    show_cdf: bool = True,
) -> Image.Image:
    """ヒストグラムを PIL 画像として描画する（value / count 軸に目盛り付き）。"""
    img = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    margin_l, margin_r, margin_t, margin_b = 52, 42, 16, 40
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    y_bottom = margin_t + plot_h

    draw.rectangle(
        [margin_l, margin_t, margin_l + plot_w, y_bottom],
        outline=(80, 80, 80),
        fill=(20, 20, 20),
    )

    series: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    if hist.get("channel") == "rgb":
        colors = {"R": (220, 80, 80), "G": (80, 200, 80), "B": (80, 120, 230)}
        for name, color in colors.items():
            series.append((hist["channels"][name]["counts"], color))
        cdf_src = hist["channels"]["R"]["cdf"]
        edges = np.asarray(hist["channels"]["R"]["edges"], dtype=np.float64)
    else:
        series.append((hist["counts"], (200, 200, 200)))
        cdf_src = hist["cdf"]
        edges = np.asarray(hist["edges"], dtype=np.float64)

    vmin = float(edges[0]) if edges.size else 0.0
    vmax = float(edges[-1]) if edges.size else 1.0
    max_count = max((float(s.max()) if s.size else 0.0) for s, _ in series) or 1.0

    for counts, color in series:
        n = len(counts)
        if n == 0:
            continue
        bar_w = max(plot_w / n, 1.0)
        for i, c in enumerate(counts):
            h = (float(c) / max_count) * plot_h
            x0 = margin_l + i * bar_w
            y0 = y_bottom - h
            draw.rectangle([x0, y0, x0 + bar_w, y_bottom], fill=color)

    if show_cdf and cdf_src is not None and len(cdf_src) > 1:
        pts = []
        n = len(cdf_src)
        for i, v in enumerate(cdf_src):
            x = margin_l + (i / max(n - 1, 1)) * plot_w
            y = y_bottom - float(v) * plot_h
            pts.append((x, y))
        if len(pts) >= 2:
            draw.line(pts, fill=(255, 200, 60), width=2)

    tick_col = (150, 150, 150)
    grid_col = (55, 55, 55)

    # value（横）軸の目盛り
    x_ticks = 5
    for k in range(x_ticks + 1):
        frac = k / x_ticks
        x = margin_l + frac * plot_w
        val = vmin + (vmax - vmin) * frac
        draw.line([(x, y_bottom), (x, y_bottom + 4)], fill=tick_col)
        if 0 < k < x_ticks:
            draw.line([(x, margin_t), (x, y_bottom)], fill=grid_col)
        label = _fmt_axis_num(val)
        tw = draw.textlength(label)
        draw.text((x - tw / 2, y_bottom + 6), label, fill=tick_col)

    # count（縦）軸の目盛り
    y_ticks = 4
    for k in range(y_ticks + 1):
        frac = k / y_ticks
        y = y_bottom - frac * plot_h
        cval = max_count * frac
        draw.line([(margin_l - 4, y), (margin_l, y)], fill=tick_col)
        if 0 < k < y_ticks:
            draw.line([(margin_l, y), (margin_l + plot_w, y)], fill=grid_col)
        label = _fmt_axis_num(cval)
        tw = draw.textlength(label)
        draw.text((margin_l - 6 - tw, y - 5), label, fill=tick_col)

    # CDF（右）軸の目盛り（0〜100%）
    if show_cdf:
        for k in range(y_ticks + 1):
            frac = k / y_ticks
            y = y_bottom - frac * plot_h
            draw.line([(margin_l + plot_w, y), (margin_l + plot_w + 4, y)], fill=(150, 130, 70))
            label = f"{int(frac * 100)}%"
            draw.text((margin_l + plot_w + 6, y - 5), label, fill=(200, 170, 90))

    draw.text((4, 2), "count", fill=(160, 160, 160))
    draw.text((width // 2 - 14, height - 14), "value", fill=(160, 160, 160))
    if show_cdf:
        draw.text((width - 60, 2), "CDF", fill=(255, 200, 60))
    return img


def render_profile_image(
    distance: np.ndarray,
    values: np.ndarray,
    width: int = 420,
    height: int = 180,
) -> Image.Image:
    """ラインプロファイルを描画する。"""
    img = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    margin_l, margin_r, margin_t, margin_b = 40, 12, 12, 28
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    draw.rectangle(
        [margin_l, margin_t, margin_l + plot_w, margin_t + plot_h],
        outline=(80, 80, 80),
        fill=(20, 20, 20),
    )
    if values.size < 2:
        return img

    dmin, dmax = float(distance.min()), float(distance.max())
    vmin, vmax = float(values.min()), float(values.max())
    if dmax <= dmin:
        dmax = dmin + 1
    if vmax <= vmin:
        vmax = vmin + 1

    pts = []
    for d, v in zip(distance, values):
        x = margin_l + (float(d) - dmin) / (dmax - dmin) * plot_w
        y = margin_t + plot_h - (float(v) - vmin) / (vmax - vmin) * plot_h
        pts.append((x, y))
    draw.line(pts, fill=(100, 200, 255), width=2)
    draw.text((4, 4), f"{vmax:.3g}", fill=(160, 160, 160))
    draw.text((4, height - 40), f"{vmin:.3g}", fill=(160, 160, 160))
    draw.text((width // 2 - 24, height - 18), "distance", fill=(160, 160, 160))
    return img


def render_fft_image(magnitude: np.ndarray, cmap: str = "inferno") -> Image.Image:
    return apply_colormap(magnitude, cmap=cmap)


def overlay_roi_on_image(
    base: Image.Image,
    *,
    rect: tuple[int, int, int, int] | None = None,
    circle: tuple[int, int, float] | None = None,
    polygon: list[tuple[int, int]] | None = None,
    line: tuple[int, int, int, int] | None = None,
    color: tuple[int, int, int] = (0, 255, 128),
) -> Image.Image:
    """ROI / ラインをオーバーレイした画像を返す。"""
    img = base.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    if rect is not None:
        x0, y0, x1, y1 = rect
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
    if circle is not None:
        cx, cy, r = circle
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=2)
    if polygon is not None and len(polygon) >= 2:
        draw.line(polygon + ([polygon[0]] if len(polygon) >= 3 else []), fill=color, width=2)
        for px, py in polygon:
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=color)
    if line is not None:
        x0, y0, x1, y1 = line
        draw.line([x0, y0, x1, y1], fill=(255, 80, 80), width=2)
        draw.ellipse([x0 - 3, y0 - 3, x0 + 3, y0 + 3], fill=(255, 80, 80))
        draw.ellipse([x1 - 3, y1 - 3, x1 + 3, y1 + 3], fill=(255, 80, 80))
    return img
