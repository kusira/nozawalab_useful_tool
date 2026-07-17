import io
import time
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import os

try:
    import torch
    import face_alignment
except Exception:
    torch = None
    face_alignment = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def resolve_torch_device() -> str:
    """CUDA が利用可能なら cuda、それ以外は cpu を返す。"""
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def describe_torch_device(device: str | None = None) -> str:
    """UI表示用のデバイス説明文字列を返す。"""
    device = device or resolve_torch_device()
    if device == "cuda" and torch is not None and torch.cuda.is_available():
        return f"GPU ({torch.cuda.get_device_name(0)})"
    return "CPU"


class FaceLandmarkCalculator:
    """
    face-alignment を用いて顔ランドマークを推定するクラス。
    - 入力は BGR ndarray（OpenCV形式）または画像バイト
    - 出力はピクセル座標と正規化座標(0-1)の両方を含む辞書
    """

    # シングルトン（クラス共有）
    _fa_singleton = None
    _fa_config = None  # (landmark_type, device)
    _yolo_singleton = None

    def __init__(self, landmark_type: str = "2D", device: Optional[str] = None, yolo_path: Optional[str] = None):
        """
        landmark_type: "2D" もしくは "3D"
        device: "cuda" / "cpu"（None の場合は自動判定）
        """
        if face_alignment is None:
            raise ImportError(
                "face-alignment がインストールされていません。"
                "requirements に face-alignment, torch, torchvision, scikit-image, scipy を追加してください。"
            )

        if device is None:
            device = resolve_torch_device()

        self.device = device
        self.landmark_type = landmark_type.upper()
        if self.landmark_type not in ("2D", "3D"):
            raise ValueError("landmark_type は '2D' または '3D' を指定してください。")

        # face-alignment のバージョン差吸収: LandmarksType の列挙名が異なる場合がある
        lt = face_alignment.LandmarksType
        def _resolve_type(name_2d: bool):
            candidates = [
                ("_2D", "_3D"),
                ("TWO_D", "THREE_D"),
                ("TWO_D_POINTS", "THREE_D_POINTS"),
            ]
            for two_d_name, three_d_name in candidates:
                two = getattr(lt, two_d_name, None)
                three = getattr(lt, three_d_name, None)
                if two is not None and three is not None:
                    return two if name_2d else three
            # 最後の手段: 最初に見つかった属性を使う
            for attr in dir(lt):
                if name_2d and ("2" in attr or "TWO" in attr):
                    return getattr(lt, attr)
                if not name_2d and ("3" in attr or "THREE" in attr):
                    return getattr(lt, attr)
            raise AttributeError("LandmarksType に対応する 2D/3D 定数が見つかりません")

        landmarks_type = _resolve_type(self.landmark_type == "2D")

        # face-alignment シングルトン
        if (FaceLandmarkCalculator._fa_singleton is None or
            FaceLandmarkCalculator._fa_config != (self.landmark_type, self.device)):
            FaceLandmarkCalculator._fa_singleton = face_alignment.FaceAlignment(
                landmarks_type,
                device=self.device,
                flip_input=False,
            )
            FaceLandmarkCalculator._fa_config = (self.landmark_type, self.device)

        # YOLO シングルトン
        if FaceLandmarkCalculator._yolo_singleton is None and YOLO is not None:
            if yolo_path is None:
                yolo_path = os.path.join(os.path.dirname(__file__), "yolov8n-face-lindevs.pt")
            if os.path.exists(yolo_path):
                try:
                    FaceLandmarkCalculator._yolo_singleton = YOLO(yolo_path)
                except Exception:
                    FaceLandmarkCalculator._yolo_singleton = None

        self.fa = FaceLandmarkCalculator._fa_singleton
        self.yolo = FaceLandmarkCalculator._yolo_singleton

        # 68点ランドマークのエッジ定義（フロント側オーバーレイ用）
        self.edges_68 = self._build_68_edges()

    @staticmethod
    def _build_68_edges() -> List[Tuple[int, int]]:
        """
        68-landmark の基本接続。ループ部位（目・口）の閉路も含む。
        参考: iBUG 300-W 68点仕様
        """
        edges = []
        # 顎ライン 0-16
        edges += [(i, i + 1) for i in range(0, 16)]
        # 右眉 17-21
        edges += [(i, i + 1) for i in range(17, 21)]
        # 左眉 22-26
        edges += [(i, i + 1) for i in range(22, 26)]
        # 鼻筋 27-30
        edges += [(i, i + 1) for i in range(27, 30)]
        # 鼻先 31-35
        edges += [(i, i + 1) for i in range(31, 35)]
        # 右目 36-41（閉路）
        edges += [(36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36)]
        # 左目 42-47（閉路）
        edges += [(42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42)]
        # 口外周 48-59（閉路）
        edges += [(48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
                  (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48)]
        # 口内周 60-67（閉路）
        edges += [(60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66),
                  (66, 67), (67, 60)]
        return edges

    @staticmethod
    def _imdecode_bgr(image_bytes: bytes) -> np.ndarray:
        """画像バイトをBGR ndarrayにデコード"""
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("画像のデコードに失敗しました。")
        return img

    def _ensure_rgb(self, img_bgr: np.ndarray) -> np.ndarray:
        """BGR -> RGB へ変換（face-alignmentはRGB想定）"""
        if img_bgr.ndim == 2:
            return cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def predict_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """画像バイトからランドマーク推定"""
        img_bgr = self._imdecode_bgr(image_bytes)
        return self.predict_from_bgr(img_bgr)

    def predict_from_bgr(self, img_bgr: np.ndarray, bbox_scale: float = 1.0) -> Dict[str, Any]:
        result, _timing = self._predict_from_bgr_internal(img_bgr, bbox_scale=bbox_scale)
        return result

    def predict_from_bgr_timed(
        self, img_bgr: np.ndarray, bbox_scale: float = 1.0
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        return self._predict_from_bgr_internal(img_bgr, bbox_scale=bbox_scale)

    def _predict_from_bgr_internal(
        self, img_bgr: np.ndarray, *, bbox_scale: float = 1.0
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        """
        BGR ndarray(OpenCV) からランドマーク推定。
        戻り値:
        {
          "width": int,
          "height": int,
          "faces": [
            {
              "landmarks": [[x, y], ...],
              "landmarks_norm": [[x/w, y/h], ...],
              "edges": [[a, b], ...],
            },
            ...
          ]
        }
        """
        if img_bgr is None or not isinstance(img_bgr, np.ndarray):
            raise ValueError("img_bgr は numpy.ndarray である必要があります。")

        try:
            from module.face_pipeline import expand_bbox_xyxy
        except ModuleNotFoundError:
            from face_pipeline import expand_bbox_xyxy

        timing = {"yolo": 0.0, "landmark": 0.0, "total": 0.0}
        started_at = time.perf_counter()

        h, w = img_bgr.shape[:2]
        img_rgb = self._ensure_rgb(img_bgr)

        preds = None
        # まず YOLO で最大顔を検出し、切り出して推定（座標をオフセット）
        if self.yolo is not None:
            try:
                yolo_started_at = time.perf_counter()
                results = self.yolo.predict(source=img_rgb, verbose=False, device=self.device)
                timing["yolo"] = time.perf_counter() - yolo_started_at
                boxes = []
                for r in results:
                    if getattr(r, 'boxes', None) is None:
                        continue
                    xyxy = r.boxes.xyxy.cpu().numpy().astype(float)
                    for x1, y1, x2, y2 in xyxy:
                        boxes.append((max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)))
                if boxes:
                    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
                    x1, y1, x2, y2 = boxes[0]
                    x1, y1, x2, y2 = expand_bbox_xyxy(x1, y1, x2, y2, bbox_scale, w, h)
                    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                    crop = img_rgb[y1i:y2i, x1i:x2i]
                    if crop.size > 0:
                        landmark_started_at = time.perf_counter()
                        pc = self.fa.get_landmarks(crop)
                        timing["landmark"] += time.perf_counter() - landmark_started_at
                        if pc is not None:
                            # オフセット加算
                            preds = [p + np.array([x1, y1, 0]) if p.shape[1] == 3 else p + np.array([x1, y1]) for p in pc]
            except Exception:
                preds = None

        # YOLOが無い/失敗時はフル画像で推定
        if preds is None:
            landmark_started_at = time.perf_counter()
            preds = self.fa.get_landmarks(img_rgb)
            timing["landmark"] += time.perf_counter() - landmark_started_at

        faces: List[Dict[str, Any]] = []
        if preds is None:
            timing["total"] = time.perf_counter() - started_at
            return {"width": w, "height": h, "faces": faces}, timing

        for pts in preds:
            # 2D: (68,2) / 3D: (68,3) を想定 → x,y は先頭2次元を利用
            xy = pts[:, :2].astype(np.float32)
            # ピクセル座標
            landmarks_px = xy.tolist()
            # 正規化
            landmarks_norm = (xy / np.array([w, h], dtype=np.float32)).tolist()

            faces.append(
                {
                    "landmarks": landmarks_px,
                    "landmarks_norm": landmarks_norm,
                    "edges": self.edges_68,
                }
            )

        timing["total"] = time.perf_counter() - started_at
        return {"width": w, "height": h, "faces": faces}, timing

    def draw_debug(self, img_bgr: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        デバッグ用：ランドマークとエッジを画像に描画して返す。
        """
        try:
            from module.face_pipeline import draw_landmarks_overlay
        except ModuleNotFoundError:
            from face_pipeline import draw_landmarks_overlay

        result = self.predict_from_bgr(img_bgr)
        return draw_landmarks_overlay(img_bgr, result, color=color)


