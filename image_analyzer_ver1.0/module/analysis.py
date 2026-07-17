"""画像の統計・ヒストグラム・品質・FFT・比較解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


COLORMAPS = (
    "gray",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "jet",
    "hot",
    "cool",
    "turbo",
)


def to_float_gray(array: np.ndarray) -> np.ndarray:
    """解析用の 2D float64 輝度配列へ変換する。"""
    arr = np.asarray(array)
    if arr.ndim == 3:
        if arr.shape[-1] >= 3:
            # ITU-R BT.601
            rgb = arr[..., :3].astype(np.float64)
            return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        return arr[..., 0].astype(np.float64)
    if arr.ndim == 1:
        side = int(np.sqrt(arr.size))
        if side * side != arr.size:
            raise ValueError("1次元配列は正方形である必要があります。")
        return arr.reshape(side, side).astype(np.float64)
    return arr.astype(np.float64)


def compute_stats(array: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    """拡張統計量を算出する。"""
    arr = np.asarray(array)
    if mask is not None:
        flat = arr[mask.astype(bool)].astype(np.float64).ravel()
    else:
        flat = arr.astype(np.float64).ravel()

    if flat.size == 0:
        return {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "nan_count": 0,
            "inf_count": 0,
            "zero_ratio": 0.0,
            "saturated_ratio": 0.0,
        }

    finite = flat[np.isfinite(flat)]
    nan_count = int(np.isnan(flat).sum())
    inf_count = int(np.isinf(flat).sum())
    if finite.size == 0:
        return {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "count": int(flat.size),
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "zero_ratio": 0.0,
            "saturated_ratio": 0.0,
        }

    vmin = float(finite.min())
    vmax = float(finite.max())
    # dtype の理論上限付近を飽和とみなす
    sat_thr = vmax
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        sat_thr = float(info.max) * 0.98
    elif vmax > 1.0:
        sat_thr = vmax * 0.98
    else:
        sat_thr = 0.98

    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "count": int(finite.size),
        "min": vmin,
        "max": vmax,
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "median": float(np.median(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p05": float(np.percentile(finite, 5)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_ratio": float((finite == 0).mean()),
        "saturated_ratio": float((finite >= sat_thr).mean()) if finite.size else 0.0,
    }


def _aligned_int_edges(
    values: np.ndarray,
    lo: float,
    hi: float,
    max_bins: int = 65536,
) -> np.ndarray | None:
    """画素値を整数とみなし、min〜max を 1 階調ずつ数えるためのビン境界を作る。

    bins で機械的に分割すると、階調数より分割数が多いとき空ビンが規則的に挟まり
    くし状（ギザギザ）になる。ここでは値を整数へ丸め、各整数値が 1 ビンに入るよう
    境界を整数グリッドに整列させる。飛び飛び（例: 偶数のみ）のデータでも、その
    量子化ステップに合わせるので空ビンが挟まらない。
    階調が多すぎる場合のみ None を返し、通常の等間隔ビンにフォールバックする。
    """
    # 画素値は整数。小数（グレー変換等）を丸めてから階調を評価する。
    rounded = np.rint(values)
    lo_i = float(np.floor(lo))
    hi_i = float(np.ceil(hi))
    in_range = rounded[(rounded >= lo_i) & (rounded <= hi_i)]
    if in_range.size == 0:
        return None
    vals, counts = np.unique(in_range, return_counts=True)
    if vals.size < 2:
        return None

    # 量子化ステップは「主要な階調の最頻間隔」で推定する。
    # 最小間隔を使うと、補間端などで稀に現れる中間値が1つあるだけで step=1 となり、
    # 実際は飛び飛び（例: 偶数のみ）の値でも空・低ビンが規則的に挟まりくし状になる。
    # 頻度が最大の一定割合以上の階調だけを見て、その最頻間隔を採用する。
    threshold = counts.max() * 0.10
    major = vals[counts >= threshold]
    ref = major if major.size >= 2 else vals
    diffs = np.diff(ref)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    gap_vals, gap_counts = np.unique(diffs, return_counts=True)
    step = float(gap_vals[int(np.argmax(gap_counts))])
    if step <= 0:
        step = 1.0
    levels = int(round((hi_i - lo_i) / step)) + 1
    if levels < 2 or levels > max_bins:
        return None
    # 各整数階調をビン中心に置くよう半ステップずらした整数グリッド。
    return (lo_i - step / 2.0) + step * np.arange(levels + 1)


def _histogram_1d(
    values: np.ndarray,
    bins: int,
    value_range: tuple[float, float] | None,
    auto_bins: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """1次元データからヒストグラム（counts, edges）を作る。value_range 指定時はその範囲のみ集計。"""
    values = values[np.isfinite(values)]
    if value_range is not None:
        lo, hi = value_range
        if hi < lo:
            lo, hi = hi, lo
        values = values[(values >= lo) & (values <= hi)]
    elif values.size:
        lo, hi = float(values.min()), float(values.max())
    else:
        lo, hi = 0.0, 1.0

    if values.size == 0:
        return np.zeros(bins, dtype=np.float64), np.linspace(lo, hi, bins + 1)
    if hi <= lo:
        hi = lo + 1e-6

    edges: np.ndarray | None = None
    if auto_bins:
        edges = _aligned_int_edges(values, lo, hi)
    if edges is not None:
        counts, out_edges = np.histogram(values, bins=edges)
    else:
        counts, out_edges = np.histogram(values, bins=bins, range=(lo, hi))
    return counts.astype(np.float64), out_edges


def compute_histogram(
    array: np.ndarray,
    bins: int = 256,
    mask: np.ndarray | None = None,
    channel: str = "gray",
    value_range: tuple[float, float] | None = None,
    auto_bins: bool = False,
) -> dict[str, Any]:
    """ヒストグラムと累積分布を算出する。

    value_range を指定するとその値域のみ集計する。auto_bins=True のときは、
    量子化された整数階調に境界を合わせてくし状アーティファクトを抑える。
    """
    arr = np.asarray(array)
    if channel == "gray" or arr.ndim == 2:
        data = to_float_gray(arr)
        if mask is not None:
            values = data[mask.astype(bool)].ravel()
        else:
            values = data.ravel()
        counts, edges = _histogram_1d(values, bins, value_range, auto_bins)
        cdf = np.cumsum(counts)
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]
        return {"channel": "gray", "counts": counts, "edges": edges, "cdf": cdf}

    # RGB 各チャンネル
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return compute_histogram(
            arr, bins=bins, mask=mask, channel="gray", value_range=value_range, auto_bins=auto_bins
        )

    result: dict[str, Any] = {"channel": "rgb", "channels": {}}
    names = ("R", "G", "B")
    for i, name in enumerate(names):
        ch = arr[..., i].astype(np.float64)
        if mask is not None:
            values = ch[mask.astype(bool)].ravel()
        else:
            values = ch.ravel()
        counts, edges = _histogram_1d(values, bins, value_range, auto_bins)
        cdf = np.cumsum(counts)
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]
        result["channels"][name] = {"counts": counts, "edges": edges, "cdf": cdf}
    return result


def extract_line_profile(
    array: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    num: int | None = None,
) -> dict[str, np.ndarray]:
    """2点間の輝度プロファイルをサンプリングする。"""
    gray = to_float_gray(array)
    h, w = gray.shape
    x0 = int(np.clip(x0, 0, w - 1))
    y0 = int(np.clip(y0, 0, h - 1))
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))

    length = float(np.hypot(x1 - x0, y1 - y0))
    if num is None:
        num = max(int(round(length)) + 1, 2)
    num = max(num, 2)

    xs = np.linspace(x0, x1, num)
    ys = np.linspace(y0, y1, num)
    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)
    values = gray[yi, xi]
    distance = np.linspace(0.0, length, num)
    return {"distance": distance, "values": values, "xs": xs, "ys": ys}


def rect_mask(shape: tuple[int, ...], x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    h, w = shape[:2]
    xa, xb = sorted((int(x0), int(x1)))
    ya, yb = sorted((int(y0), int(y1)))
    xa = max(0, min(xa, w - 1))
    xb = max(0, min(xb, w - 1))
    ya = max(0, min(ya, h - 1))
    yb = max(0, min(yb, h - 1))
    mask = np.zeros((h, w), dtype=bool)
    mask[ya : yb + 1, xa : xb + 1] = True
    return mask


def circle_mask(shape: tuple[int, ...], cx: int, cy: int, radius: float) -> np.ndarray:
    h, w = shape[:2]
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def polygon_mask(shape: tuple[int, ...], points: list[tuple[int, int]]) -> np.ndarray:
    """多角形 ROI のマスク（スキャンライン）。"""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    if len(points) < 3:
        return mask
    pts = np.asarray(points, dtype=np.float64)
    min_y = max(0, int(np.floor(pts[:, 1].min())))
    max_y = min(h - 1, int(np.ceil(pts[:, 1].max())))
    n = len(pts)
    for y in range(min_y, max_y + 1):
        nodes: list[float] = []
        j = n - 1
        for i in range(n):
            yi, yj = pts[i, 1], pts[j, 1]
            xi, xj = pts[i, 0], pts[j, 0]
            if (yi < y and yj >= y) or (yj < y and yi >= y):
                if yj != yi:
                    nodes.append(xi + (y - yi) / (yj - yi) * (xj - xi))
            j = i
        nodes.sort()
        for k in range(0, len(nodes) - 1, 2):
            xa = max(0, int(np.ceil(nodes[k])))
            xb = min(w - 1, int(np.floor(nodes[k + 1])))
            if xa <= xb:
                mask[y, xa : xb + 1] = True
    return mask


@dataclass
class QualityResult:
    blur_score: float
    blur_flag: bool
    underexposed_ratio: float
    overexposed_ratio: float
    underexposed_flag: bool
    overexposed_flag: bool
    noise_estimate: float
    noise_flag: bool
    resolution_ok: bool
    notes: list[str]


def assess_quality(
    array: np.ndarray,
    *,
    blur_threshold: float = 80.0,
    under_thr: float = 0.05,
    over_thr: float = 0.95,
    extreme_ratio_thr: float = 0.05,
    noise_threshold: float = 15.0,
    min_side: int = 32,
) -> QualityResult:
    """ぼけ・露出・ノイズ・解像度の簡易品質チェック。"""
    gray = to_float_gray(array)
    h, w = gray.shape
    notes: list[str] = []

    # 表示用に 0-255 正規化してから判定
    g = gray.copy()
    gmin, gmax = float(np.nanmin(g)), float(np.nanmax(g))
    if gmax > gmin:
        g_norm = (g - gmin) / (gmax - gmin) * 255.0
    else:
        g_norm = np.zeros_like(g)

    # Laplacian 分散（ぼけ）
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    # 簡易畳み込み
    padded = np.pad(g_norm, 1, mode="edge")
    lap = (
        kernel[0, 1] * padded[0:-2, 1:-1]
        + kernel[1, 0] * padded[1:-1, 0:-2]
        + kernel[1, 1] * padded[1:-1, 1:-1]
        + kernel[1, 2] * padded[1:-1, 2:]
        + kernel[2, 1] * padded[2:, 1:-1]
    )
    blur_score = float(lap.var())
    blur_flag = blur_score < blur_threshold
    if blur_flag:
        notes.append(f"ぼけの可能性（Laplacian分散={blur_score:.1f} < {blur_threshold}）")

    under_ratio = float((g_norm <= under_thr * 255).mean())
    over_ratio = float((g_norm >= over_thr * 255).mean())
    under_flag = under_ratio >= extreme_ratio_thr
    over_flag = over_ratio >= extreme_ratio_thr
    if under_flag:
        notes.append(f"露出不足の可能性（暗い画素 {under_ratio * 100:.1f}%）")
    if over_flag:
        notes.append(f"露出過多の可能性（明るい画素 {over_ratio * 100:.1f}%）")

    # ノイズ推定: 局所差分の MAD 近似
    dx = np.diff(g_norm, axis=1)
    dy = np.diff(g_norm, axis=0)
    mad = float(np.median(np.abs(np.concatenate([dx.ravel(), dy.ravel()]))))
    noise_est = mad * 1.4826
    noise_flag = noise_est > noise_threshold
    if noise_flag:
        notes.append(f"ノイズが大きい可能性（推定={noise_est:.1f} > {noise_threshold}）")

    resolution_ok = h >= min_side and w >= min_side
    if not resolution_ok:
        notes.append(f"解像度が低い（{w}×{h}、最小辺 {min_side} 推奨）")

    if not notes:
        notes.append("特記事項なし（閾値内）")

    return QualityResult(
        blur_score=blur_score,
        blur_flag=blur_flag,
        underexposed_ratio=under_ratio,
        overexposed_ratio=over_ratio,
        underexposed_flag=under_flag,
        overexposed_flag=over_flag,
        noise_estimate=noise_est,
        noise_flag=noise_flag,
        resolution_ok=resolution_ok,
        notes=notes,
    )


def compute_fft_magnitude(array: np.ndarray) -> np.ndarray:
    """中心化した FFT 振幅スペクトル（log1p）。"""
    gray = to_float_gray(array)
    windowed = gray - float(np.mean(gray))
    spec = np.fft.fftshift(np.fft.fft2(windowed))
    mag = np.log1p(np.abs(spec))
    return mag


def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff**2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float | None = None) -> float:
    err = mse(a, b)
    if err <= 0:
        return float("inf")
    if data_range is None:
        data_range = float(max(np.max(a), np.max(b)) - min(np.min(a), np.min(b)))
        if data_range <= 0:
            data_range = 1.0
    return float(10.0 * np.log10((data_range**2) / err))


def _resize_max_side(arr: np.ndarray, max_side: int = 256) -> np.ndarray:
    """SSIM 高速化のため長辺を max_side 以下に縮小する（対応ブロック平均）。"""
    h, w = arr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 0.999:
        return arr
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    if nh == h and nw == w:
        return arr

    y0 = (np.arange(nh) * h) // nh
    y1 = np.maximum(((np.arange(nh) + 1) * h) // nh, y0 + 1)
    x0 = (np.arange(nw) * w) // nw
    x1 = np.maximum(((np.arange(nw) + 1) * w) // nw, x0 + 1)

    src = arr.astype(np.float64, copy=False)
    padded = np.zeros((h + 1, w + 1), dtype=np.float64)
    padded[1:, 1:] = src
    integral = padded.cumsum(0).cumsum(1)
    iy0, iy1 = y0[:, None], y1[:, None]
    ix0, ix1 = x0[None, :], x1[None, :]
    sums = integral[iy1, ix1] - integral[iy0, ix1] - integral[iy1, ix0] + integral[iy0, ix0]
    area = (iy1 - iy0).astype(np.float64) * (ix1 - ix0).astype(np.float64)
    return sums / np.maximum(area, 1.0)


def ssim(a: np.ndarray, b: np.ndarray, data_range: float | None = None) -> float:
    """簡易 SSIM（縮小後に 11x11 ガウシアン窓で算出）。"""
    x = _resize_max_side(a.astype(np.float64))
    y = _resize_max_side(b.astype(np.float64))
    if data_range is None:
        data_range = float(max(x.max(), y.max()) - min(x.min(), y.min()))
        if data_range <= 0:
            data_range = 1.0

    size = 11
    sigma = 1.5
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    def conv2(img: np.ndarray) -> np.ndarray:
        pad = size // 2
        padded = np.pad(img, pad, mode="reflect")
        out = np.zeros_like(img)
        for i in range(size):
            for j in range(size):
                out += kernel[i, j] * padded[i : i + img.shape[0], j : j + img.shape[1]]
        return out

    mu_x = conv2(x)
    mu_y = conv2(y)
    mu_x2 = mu_x**2
    mu_y2 = mu_y**2
    mu_xy = mu_x * mu_y
    sigma_x2 = conv2(x**2) - mu_x2
    sigma_y2 = conv2(y**2) - mu_y2
    sigma_xy = conv2(x * y) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    map_ssim = num / (den + 1e-12)
    return float(map_ssim.mean())


def compare_images(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """2画像を同サイズに合わせて比較指標を算出する。"""
    ga = to_float_gray(a)
    gb = to_float_gray(b)
    h = min(ga.shape[0], gb.shape[0])
    w = min(ga.shape[1], gb.shape[1])
    ga = ga[:h, :w]
    gb = gb[:h, :w]
    abs_diff = np.abs(ga - gb)
    return {
        "mse": mse(ga, gb),
        "mae": mae(ga, gb),
        "psnr": psnr(ga, gb),
        "ssim": ssim(ga, gb),
        "abs_diff": abs_diff,
        "signed_diff": ga - gb,
        "shape": (h, w),
    }


def stats_row_for_export(path: Path, array: np.ndarray, quality: QualityResult | None = None) -> dict[str, Any]:
    """バッチ出力用の1行辞書。"""
    st = compute_stats(array)
    row: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "shape": str(st["shape"]),
        "dtype": st["dtype"],
        "count": st["count"],
        "min": st["min"],
        "max": st["max"],
        "mean": st["mean"],
        "std": st["std"],
        "median": st["median"],
        "p01": st["p01"],
        "p05": st["p05"],
        "p25": st["p25"],
        "p75": st["p75"],
        "p95": st["p95"],
        "p99": st["p99"],
        "nan_count": st["nan_count"],
        "inf_count": st["inf_count"],
        "zero_ratio": st["zero_ratio"],
        "saturated_ratio": st["saturated_ratio"],
    }
    if quality is not None:
        row.update(
            {
                "blur_score": quality.blur_score,
                "blur_flag": quality.blur_flag,
                "underexposed_ratio": quality.underexposed_ratio,
                "overexposed_ratio": quality.overexposed_ratio,
                "noise_estimate": quality.noise_estimate,
                "noise_flag": quality.noise_flag,
                "resolution_ok": quality.resolution_ok,
                "quality_notes": "; ".join(quality.notes),
            }
        )
    return row
