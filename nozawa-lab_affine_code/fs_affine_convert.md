# fs_affine_convert 使い方

顔の 68 点ランドマーク（dlib 68 点モデル相当）を使い、**Piecewise Affine Warp**（三角形ごとのアフィン変換）で顔画像を標準テンプレートに整列するモジュールです。

---

## 概要

| 項目 | 内容 |
|------|------|
| ファイル | `fs_affine_convert.py` |
| 主クラス | `affine_convert` |
| 入力 | 画像（BGR またはグレースケール）＋ 68 点ランドマーク |
| 出力 | アフィン変換後の画像（`afsize` で指定したサイズ） |
| 依存 | OpenCV (`opencv-python`), NumPy |

ランドマーク検出そのものは行いません。dlib / face-alignment など別モジュールで 68 点を取得してから本モジュールに渡してください。

---

## クイックスタート

```python
import cv2
import numpy as np
from fs_affine_convert import affine_convert

# 画像読み込み（BGR）
src = cv2.imread("face.png")

# 68 点ランドマーク（例: dlib / face-alignment の出力）
# landmarks[i] = [x, y] 形式、インデックス 0〜67
landmarks = np.array([...], dtype=np.float64)  # shape: (68, 2)

# アフィン変換器を作成
converter = affine_convert(afsize=[256, 256], square=True)

# 変換実行
result = converter.main(src, landmarks)

cv2.imwrite("affine_result.png", result)
```

---

## クラス `affine_convert`

### コンストラクタ

```python
affine_convert(afsize, square, temp_parts=None)
```

| 引数 | 型 | 説明 |
|------|-----|------|
| `afsize` | `list[int]` | 出力画像サイズ `[高さ, 幅]`。例: `[256, 256]` |
| `square` | `bool` | `True`: 顔輪郭を四角形に整えて変換 / `False`: 顔形状のまま変換 |
| `temp_parts` | `np.ndarray` または `None` | テンプレートランドマーク `(N, 2)`。`None` のとき `MEAN_FRONT_PARTS_LIST` を使用 |

### メソッド `main`

```python
result = converter.main(src, landmark, binary_on=False)
```

| 引数 | 型 | 説明 |
|------|-----|------|
| `src` | `np.ndarray` | 入力画像。BGR `(H, W, 3)` またはグレースケール `(H, W)` |
| `landmark` | `np.ndarray` | 68 点ランドマーク `(68, 2)`。各点は `[x, y]` |
| `binary_on` | `bool` | サーモグラフィなど **1 チャンネル数値データ** 向け。`square=True` 時のみ有効 |

**戻り値:** `np.ndarray` — BGR 画像 `(afsize[0], afsize[1], 3)`。`binary_on=True` のときはグレースケール `(afsize[0], afsize[1])`。

---

## アフィンサイズ（`afsize`）について

### 基本

`afsize` は **出力画像のピクセルサイズ** を `[高さ, 幅]` の順で指定します。

```python
afsize = [256, 256]   # 256 × 256（正方形）
afsize = [120, 120]   # 120 × 120（正方形）
afsize = [200, 256]   # 高さ 200 × 幅 256（長方形も指定可能）
```

内部では次のように使われます。

```python
dst = np.zeros((afsize[0], afsize[1]))  # 高さ × 幅
```

### デフォルトテンプレートとの関係

`MEAN_FRONT_PARTS_LIST`（`default_mean_front_parts()`）は、従来の **259×255 座標系**（`mean_front_parts_list.npy` 由来）から **直接 256×256** に変換した座標です。

```
x₂₅₆ = round(x₂₅₉ × 256 / 259)
y₂₅₆ = round(y₂₅₅ × 256 / 255)
```

x・y でスケール係数が異なるため、元の縦横比が保たれます（250×250 経由の2段階変換とは結果が異なります）。座標の最大値はおおよそ 256 です。

| `square` | テンプレートの扱い | 推奨 `afsize` |
|----------|-------------------|---------------|
| `False` | テンプレート座標をそのまま使用（スケールなし） | **`[256, 256]`**（テンプレートと同じ座標系） |
| `True` | `shiftTempParts()` で輪郭を四角化し、`afsize` に合わせてスケール | **任意のサイズ**（例: `[120, 120]`, `[256, 256]`） |

`square=False` で `afsize` を 256 以外にすると、テンプレート座標と出力キャンバスのサイズが一致せず、顔がはみ出したり欠けたりする可能性があります。

### `square=True` 時のスケーリング

`square=True` のとき、コンストラクタ内で `shiftTempParts()` が呼ばれ、テンプレート座標が次のように変換されます。

1. 顔輪郭（インデックス 0〜26）を四角形に再配置
2. 鼻・目・口（インデックス 27〜67）を下方向へオフセット
3. 座標を `afsize` に正規化（最大座標が `afsize[0]-1`, `afsize[1]-1` になるようスケール）

そのため **`square=True` なら `afsize` を自由に変更できます**。小さいサイズ（例: 120×120）でもテンプレートが自動で合わせられます。

### よく使うサイズ例

| 用途 | `afsize` | `square` | 備考 |
|------|----------|----------|------|
| param_affine_tool 既定 | `[256, 256]` | `True` | GUI ツールのデフォルト出力 |
| 軽量・プレビュー用 | `[120, 120]` | `True` | 処理速度・メモリ削減 |
| 顔形状を保持した整列 | `[256, 256]` | `False` | 輪郭の自然な形状を維持 |
| サーモ温度マップ | `[256, 256]` | `True` | `binary_on=True` で 1ch データ対応 |

### 長方形出力について

`afsize` 自体は `[高さ, 幅]` で長方形を指定できますが、**最終的に任意の縦横比にしたい場合**は、変換後に `cv2.resize` でリサイズする方法が確実です（`param_affine_tool` の `warp_face_fs_affine` も同様の考え方です）。

```python
# 内部は max(幅, 高さ) の正方形で変換してからリサイズ
max_size = max(target_w, target_h)
converter = affine_convert(afsize=[max_size, max_size], square=True)
result = converter.main(src, landmarks)
result = cv2.resize(result, (target_w, target_h))
```

---

## `square` パラメータ

### `square=True`（推奨・デフォルト運用）

- 顔輪郭を **四角形** に変形してからテンプレートへマッピング
- 三角形分割は `triangle_parts_list_square` を使用
- 出力がキャンバスいっぱいに収まりやすく、機械学習の入力画像として扱いやすい

### `square=False`

- 顔の **自然な輪郭形状** を保ったまま整列
- 三角形分割は `triangle_parts_list` を使用
- テンプレート座標はスケールされないため、`afsize=[256, 256]` で使うのが安全

---

## ランドマークについて

### 必要な点数

**68 点**（インデックス 0〜67）が必須です。dlib の `shape_predictor_68_face_landmarks` や face-alignment の 68 点出力と互換です。

### テンプレート座標 `MEAN_FRONT_PARTS_LIST`

平均正面顔の 68 点座標がモジュール内に定義されています。元座標は 259（幅）× 255（高さ）のテンプレートから直接 256×256 にスケールしたものです。カスタムテンプレートを使う場合は `temp_parts` に `(68, 2)` の `numpy` 配列を渡します。

```python
from fs_affine_convert import default_mean_front_parts

template = default_mean_front_parts()  # shape: (68, 2), dtype: int32
converter = affine_convert([256, 256], square=True, temp_parts=template)
```

---

## サーモグラフィ・温度データ（`binary_on`）

1 チャンネルの数値配列（温度値など）を変換するときは `binary_on=True` を指定します。

```python
thermal = np.load("temperature.npy")  # shape: (H, W), float など

converter = affine_convert(afsize=[256, 256], square=True)
result = converter.main(thermal, landmarks, binary_on=True)
# result: shape (256, 256), グレースケール（1ch）
```

- `square=True` かつ `binary_on=True` のときのみ、内部で一時的に 3ch に変換して処理後 1ch に戻します
- `square=False` のとき `binary_on` は想定外の動作になる可能性があるため、温度データでは `square=True` を推奨

---

## 処理の流れ

```
入力画像 (src) + 68点ランドマーク (landmark)
        ↓
  afsize サイズの空キャンバス (dst) を生成
        ↓
  顔メッシュを三角形に分割
        ↓
  各三角形について
    ・元画像の 3 点 ↔ テンプレートの 3 点
    ・アフィン変換行列を計算 (cv2.getAffineTransform)
    ・三角形領域をワープして dst に合成
        ↓
  アフィン変換済み画像を返却
```

---

## 使用例

### 例 1: 標準的な顔画像（256×256・四角変換）

```python
import cv2
import numpy as np
from fs_affine_convert import affine_convert

src = cv2.imread("input.png")
landmarks = np.load("landmarks.npy")  # (68, 2)

converter = affine_convert(afsize=[256, 256], square=True)
out = converter.main(src, landmarks)

cv2.imwrite("output_256.png", out)
```

### 例 2: 小さいアフィンサイズ（120×120）

```python
converter = affine_convert(afsize=[120, 120], square=True)
out = converter.main(src, landmarks)
# 出力: (120, 120, 3)
```

### 例 3: 顔形状を保持（square=False）

```python
converter = affine_convert(afsize=[256, 256], square=False)
out = converter.main(src, landmarks)
```

### 例 4: コンバータの使い回し

`affine_convert` のインスタンスは設定（`afsize`, `square`, `temp_parts`）が同じなら **複数画像に再利用** できます。テンプレートの前処理はコンストラクタで一度だけ行われます。

```python
converter = affine_convert([256, 256], True)

for img, lm in zip(images, landmarks_list):
    results.append(converter.main(img, np.array(lm)))
```

---

## 補助関数・メソッド

| 名前 | 説明 |
|------|------|
| `default_mean_front_parts()` | デフォルトテンプレート `(68, 2)` を返す |
| `warp_triangle()` | 3 点指定のアフィン変換（内部利用） |
| `warp_rectangle()` | 4 点指定の透視変換（内部利用） |
| `shiftTempParts()` | テンプレートを四角化・スケール（`square=True` 時に自動実行） |

---

## 注意事項

- ランドマークは **68 点未満** だとインデックスエラーになります
- ランドマークの座標は **入力画像 `src` と同じピクセル座標系** である必要があります
- `square=False` のときは `afsize` をテンプレート座標系（256×256）に合わせてください
- 出力の dtype は入力処理に依存します。表示・保存前に `astype(np.uint8)` が必要な場合があります
- 背景は黒（0）で初期化されます。変換されなかった領域は黒のまま残ります

---

## param_affine_tool との関係

[`param_affine_tool_ver1.0`](../param_affine_tool_ver1.0/) では、本モジュールと同一の `fs_affine_convert.py` を `module/` 配下に置き、`face_pipeline.warp_face_fs_affine()` 経由で呼び出しています。GUI ツールのデフォルト出力は **256 × 256・square=True** です。

スタンドアロンで使う場合は、このディレクトリの `fs_affine_convert.py` を直接 import してください。
