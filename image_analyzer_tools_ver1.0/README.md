作成者: 山口高璃
github: [http://github.com/kusira/nozawalab_useful_tool/tree/main/image_analyzer_tools_ver1.0](http://github.com/kusira/nozawalab_useful_tool/tree/main/image_analyzer_tools_ver1.0)

# image_analyzer_tools

野沢研の画像ツールをまとめたスイートです。共通ライブラリ（`common/`）を土台に、
2 つのアプリを収録しています。

- **image_studio** … 画像ビューワー（旧 `image_viewer`）と画像解析（旧 `image_analyzer`）を統合したアプリ
- **param_affine_tool** … 顔特徴点（face-alignment / dlib）算出とアフィン変換を行うアプリ

**Version: 1.0**
**推奨OS: Windows**

---

## ディレクトリ構成

```
image_analyzer_tools_ver1.0/
├── README.md
├── requirements.txt
├── common/                     # 共通ライブラリ（各アプリから import）
│   ├── __init__.py
│   ├── constants.py            # RAWデータ型・リサイズ手法・保存先など
│   ├── image_loader.py         # NPY / RAW / ラスター画像の読み込み
│   ├── file_tree.py            # 階層ファイル一覧（D0001 / 0001 番号体系）
│   └── dialogs.py              # RAW 読み込み設定ダイアログ
│
├── image_studio/               # ビューワー + 解析の統合アプリ
│   ├── main.py
│   ├── startup_image_studio.vbs
│   └── module/
│       ├── app_ui.py           # メニュー・ツールバー・3パネル・解析タブ
│       ├── app_navigation.py   # ファイル操作・ツリー・移動・読み込み
│       ├── app_view.py         # 表示・リサイズ・画像処理・ルーペ・ROI/ライン
│       ├── app_analysis.py     # 統計/ヒスト/ROI/プロファイル/品質/FFT/比較
│       ├── app_export.py       # 画像/解析画像の保存・バッチ CSV/JSON 出力
│       ├── analysis.py         # 数値解析コア
│       ├── visualization.py    # カラーマップ・プロット描画
│       ├── resize.py           # 配列リサイズ
│       ├── image_processing.py # 表示画像の見た目調整
│       └── canvas.py           # ズーム/パン/ROI/ライン/ルーペ対応キャンバス
│
└── param_affine_tool/          # 顔特徴点・アフィン変換アプリ
    ├── main.py
    ├── startup_param_affine_tool.vbs
    └── module/
        ├── app_ui.py / app_navigation.py / app_image_io.py
        ├── app_crop.py / app_preview.py / app_landmark.py
        ├── face_pipeline.py / fa_landmark_calculator.py
        ├── dlib_landmark_calclator.py / fs_affine_convert.py
        ├── file_tree.py         # 対応拡張子が狭いため専用（NPY/RAW/PNG/JPEG）
        ├── shape_predictor_68_face_landmarks.dat  # dlib モデル
        └── yolov8n-face-lindevs.pt                # YOLO 顔検出モデル
```

`common/` は各アプリの `main.py` がスイートのルートを `sys.path` に追加することで
`common.xxx` として import されます。`image_studio` は `common` の 4 モジュールを、
`param_affine_tool` は `common.constants` と `common.dialogs` を利用します
（対応拡張子が異なるため、`param_affine_tool` の `file_tree` は専用のものを使います）。

---

## 起動方法

各アプリのフォルダにある VBS をダブルクリックすると、コマンドプロンプトを開かずに起動します。

| アプリ | 起動ファイル |
|--------|--------------|
| image_studio | `image_studio/startup_image_studio.vbs` |
| param_affine_tool | `param_affine_tool/startup_param_affine_tool.vbs` |

コマンドラインからも起動できます（各アプリのフォルダで実行）。

```bash
python image_studio/main.py
python param_affine_tool/main.py
```

`image_studio` は起動時にパス（ファイル／フォルダ）を指定できます。

```bash
python image_studio/main.py C:\path\to\images
```

---

## image_studio（ビューワー + 解析）

ディレクトリ／単体ファイルから画像を読み込み、表示・画像処理・ルーペ・各種解析・
エクスポートを 1 つのウィンドウで行います。

### 対応ファイル形式
`.npy` / `.raw` / `.png` / `.jpg` / `.jpeg` / `.bmp` / `.tif` / `.tiff` / `.webp`

### 画面構成
- **左**: ファイル一覧（階層ツリー）＋ 統計情報
- **中央**: プレビュー（ズーム・パン・ROI/ライン描画）＋ 反転/回転ボタン
- **右**: タブ（調整 / ヒストグラム / ROI / プロファイル / 比較 / 品質 / FFT / バッチ）
- ツールバー: ファイル操作・移動・Fit/100%・リサイズ・表示モード・カラーマップ

### 解析（生データに対して実行）
- 拡張統計（min/max/mean/std/median/パーセンタイル/ゼロ率/飽和率/NaN・Inf）
- ヒストグラム（gray/RGB、CDF、値域指定、ROI 内集計、bins 自動調整）
- ROI（矩形 / 円 / 自由選択）とラインプロファイル
- 2 枚比較（MSE / MAE / PSNR / SSIM、差分ヒートマップ）
- 品質チェック（ぼけ / 露出 / ノイズ / 解像度）
- FFT（振幅スペクトル）
- バッチ解析（一覧すべてを CSV / JSON 出力）

### 表示・調整（見た目のみ／ビューワー由来）
- リサイズ（0.1x〜1.0x、6 手法。解析対象にも反映）
- 表示モード（通常 / カラーマップ / FFT / 差分ヒートマップ）
- 「調整」タブの画像処理スライダー（明るさ・コントラスト・ガンマ・クリップ・ぼかし・
  シャープ・二値化・均等化・反転）… **表示の見た目のみ**を変更し、解析結果には影響しません
- ルーペ（拡大表示）、左右/上下反転・90° 回転
- 処理後画像のエクスポート（PNG / JPEG / NPY）

### ショートカット
`Ctrl+O` 開く / `Ctrl+Shift+O` フォルダ / `←``→` 前後 / `R` ランダム /
`Home`/`End` 先頭・末尾 / `F` Fit / `+`/`-` 拡大縮小 / `0` 100% / `Esc` ルーペを閉じる /
パン: 中クリック・右ドラッグ・Ctrl+左ドラッグ

---

## param_affine_tool（顔特徴点・アフィン変換）

NPY / RAW / PNG / JPEG を読み込み、反転・トリミング・リサイズ・画像処理を行い、
face-alignment（FA）と dlib で顔特徴点を算出して 256×256 にアフィン変換します。
顔検出には YOLO を併用します。

詳しい使い方は従来の README（旧 `param_affine_tool_ver1.0`）と同じです。主な流れ:

1. ファイル/フォルダを開く
2. 反転・トリミング・リサイズ・画像処理を調整
3. FA / dlib を選んで「特徴点算出」
4. 「プレビュー保存」で各段階の画像を PNG 出力

GPU（CUDA）が使える場合、FA と YOLO は GPU で動作します（dlib は CPU）。
モデルファイル（`.dat` / `.pt`）は `param_affine_tool/module/` に同梱しています。

---

## 依存関係

`common/` と `image_studio` は基本ライブラリのみで動作します。

```
numpy>=1.24
Pillow>=10.0
```

`param_affine_tool` の特徴点算出には、追加で OpenCV / PyTorch / face-alignment /
dlib / ultralytics などが必要です（詳細は [requirements.txt](requirements.txt)）。

- Python 3.10 以上を推奨

---

## 旧バージョンからの変更点

- `image_viewer` と `image_analyzer` を **`image_studio`** に統合しました。
  - 解析（統計・ヒスト・ROI・FFT など）は生データに対して実行します。
  - ビューワー由来の画像処理スライダー・ルーペ・反転/回転・画像エクスポートを「調整」タブ等に統合しました。
- 3 アプリで重複していたファイル読み込み・階層ツリー・RAW ダイアログ・定数を
  **`common/` 共通ライブラリ**に集約し、メンテナンスを一元化しました。
