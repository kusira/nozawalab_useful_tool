import os
import time
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

try:
    import dlib
except Exception as e:
    dlib = None

try:
    from ultralytics import YOLO  # YOLOv8
except Exception:
    YOLO = None

try:
    from module.fa_landmark_calculator import resolve_torch_device
except ModuleNotFoundError:
    from fa_landmark_calculator import resolve_torch_device


class DlibLandmarkCalculator:
    """
    dlib を用いた 68 点ランドマーク推定器。
    - 入力: BGR ndarray (OpenCV)
    - 出力: width/height と faces[ { landmarks, landmarks_norm, edges } ]
    - 複数顔時は bbox 面積最大の顔のみ返却
    """

    # シングルトン（クラス共有）
    _predictor_singleton: Optional["dlib.shape_predictor"] = None
    _detector_singleton = None
    _yolo_singleton = None

    def __init__(self, model_path: str = None, yolo_path: Optional[str] = None, device: Optional[str] = None):
        if dlib is None:
            raise ImportError("dlib がインストールされていません。")

        self.device = device if device is not None else resolve_torch_device()

        if model_path is None:
            # 同ディレクトリの shape_predictor_68_face_landmarks.dat
            model_path = os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"shape predictor が見つかりません: {model_path}")

        # dlib シングルトン
        if DlibLandmarkCalculator._detector_singleton is None:
            DlibLandmarkCalculator._detector_singleton = dlib.get_frontal_face_detector()
        if DlibLandmarkCalculator._predictor_singleton is None:
            DlibLandmarkCalculator._predictor_singleton = dlib.shape_predictor(model_path)

        # YOLO シングルトン
        if DlibLandmarkCalculator._yolo_singleton is None and YOLO is not None:
            if yolo_path is None:
                yolo_path = os.path.join(os.path.dirname(__file__), "yolov8n-face-lindevs.pt")
            if os.path.exists(yolo_path):
                try:
                    DlibLandmarkCalculator._yolo_singleton = YOLO(yolo_path)
                except Exception:
                    DlibLandmarkCalculator._yolo_singleton = None

        self.detector = DlibLandmarkCalculator._detector_singleton
        self.predictor = DlibLandmarkCalculator._predictor_singleton
        self.yolo = DlibLandmarkCalculator._yolo_singleton
        self.edges_68 = self._build_68_edges()

    @staticmethod
    def _build_68_edges() -> List[Tuple[int, int]]:
        edges = []
        edges += [(i, i + 1) for i in range(0, 16)]          # 顎 0-16
        edges += [(i, i + 1) for i in range(17, 21)]         # 右眉
        edges += [(i, i + 1) for i in range(22, 26)]         # 左眉
        edges += [(i, i + 1) for i in range(27, 30)]         # 鼻筋
        edges += [(i, i + 1) for i in range(31, 35)]         # 鼻先
        edges += [(36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36)]  # 右目
        edges += [(42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42)]  # 左目
        edges += [(48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
                  (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48)]  # 口外
        edges += [(60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66),
                  (66, 67), (67, 60)]                                          # 口内
        return edges

    def _ensure_gray(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr.ndim == 2:
            return img_bgr
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

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
        if img_bgr is None or not isinstance(img_bgr, np.ndarray):
            raise ValueError("img_bgr は numpy.ndarray である必要があります。")

        try:
            from module.face_pipeline import expand_bbox_xyxy
        except ModuleNotFoundError:
            from face_pipeline import expand_bbox_xyxy

        timing = {"yolo": 0.0, "landmark": 0.0, "total": 0.0}
        started_at = time.perf_counter()

        h, w = img_bgr.shape[:2]
        img_gray = self._ensure_gray(img_bgr)

        rects = []
        # まず YOLO で顔検出（あれば）
        if self.yolo is not None:
            try:
                yolo_started_at = time.perf_counter()
                # YOLO は RGB 前提が多いが、このモデルはBGRでも動くことが多い
                results = self.yolo.predict(source=img_bgr, verbose=False, device=self.device)
                timing["yolo"] = time.perf_counter() - yolo_started_at
                boxes = []
                for r in results:
                    if getattr(r, 'boxes', None) is None:
                        continue
                    xyxy = r.boxes.xyxy.cpu().numpy().astype(float)
                    for x1, y1, x2, y2 in xyxy:
                        boxes.append((max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)))
                if boxes:
                    # 最大面積を選択
                    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
                    x1, y1, x2, y2 = boxes[0]
                    x1, y1, x2, y2 = expand_bbox_xyxy(x1, y1, x2, y2, bbox_scale, w, h)
                    rects = [dlib.rectangle(int(x1), int(y1), int(x2), int(y2))]
            except Exception:
                rects = []

        # YOLOが無い/失敗した場合はdlibのHOG検出
        if not rects:
            landmark_started_at = time.perf_counter()
            rects = self.detector(img_gray, 1)  # upsample=1（精度と速度のバランス）
            timing["landmark"] += time.perf_counter() - landmark_started_at

        faces: List[Dict[str, Any]] = []
        for rect in rects:
            landmark_started_at = time.perf_counter()
            shape = self.predictor(img_gray, rect)
            timing["landmark"] += time.perf_counter() - landmark_started_at
            pts = [(float(shape.part(i).x), float(shape.part(i).y)) for i in range(68)]
            pts_np = np.array(pts, dtype=np.float32)

            landmarks_px = pts_np.tolist()
            landmarks_norm = (pts_np / np.array([w, h], dtype=np.float32)).tolist()

            faces.append({
                "landmarks": landmarks_px,
                "landmarks_norm": landmarks_norm,
                "edges": self.edges_68,
            })

        # bbox 最大の顔のみ
        if len(faces) > 1:
            def _bbox_area(face: Dict[str, Any]) -> float:
                pts = face.get("landmarks", [])
                if not pts:
                    return -1.0
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                return max(0.0, (max(xs) - min(xs))) * max(0.0, (max(ys) - min(ys)))

            faces.sort(key=_bbox_area, reverse=True)
            faces = [faces[0]]

        timing["total"] = time.perf_counter() - started_at
        return {"width": w, "height": h, "faces": faces}, timing

    def draw_debug(self, img_bgr: np.ndarray, color: Tuple[int, int, int] = (255, 128, 0)) -> np.ndarray:
        """デバッグ用：ランドマークとエッジを画像に描画して返す。"""
        try:
            from module.face_pipeline import draw_landmarks_overlay
        except ModuleNotFoundError:
            from face_pipeline import draw_landmarks_overlay

        result = self.predict_from_bgr(img_bgr)
        return draw_landmarks_overlay(img_bgr, result, color=color)


