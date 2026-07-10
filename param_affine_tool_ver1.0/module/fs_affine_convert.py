import cv2
import numpy as np

# dlib 68点モデル相当の (x, y) 座標（259x255 座標系から直接 256x256 に変換）
# x = round(x_259 * 256 / 259), y = round(y_255 * 256 / 255)
MEAN_FRONT_PARTS_LIST: list[list[int]] = [
    [0, 50],
    [2, 88],
    [9, 130],
    [16, 163],
    [29, 194],
    [48, 219],
    [72, 239],
    [99, 252],
    [130, 256],
    [162, 252],
    [190, 238],
    [213, 217],
    [232, 191],
    [243, 162],
    [249, 129],
    [254, 90],
    [256, 53],
    [20, 15],
    [36, 5],
    [54, 0],
    [74, 1],
    [94, 7],
    [157, 8],
    [177, 1],
    [198, 0],
    [216, 3],
    [231, 13],
    [128, 30],
    [128, 49],
    [128, 70],
    [129, 91],
    [97, 114],
    [112, 115],
    [128, 118],
    [144, 115],
    [158, 115],
    [43, 39],
    [58, 29],
    [78, 27],
    [92, 36],
    [78, 44],
    [58, 45],
    [165, 37],
    [181, 28],
    [199, 28],
    [212, 38],
    [198, 44],
    [180, 44],
    [83, 164],
    [96, 152],
    [113, 145],
    [129, 146],
    [145, 145],
    [161, 151],
    [174, 163],
    [163, 175],
    [147, 181],
    [129, 183],
    [112, 182],
    [94, 175],
    [87, 164],
    [107, 161],
    [128, 162],
    [148, 160],
    [169, 163],
    [148, 162],
    [129, 164],
    [107, 163],
]


def default_mean_front_parts() -> np.ndarray:
    """デフォルトのテンプレートランドマーク座標を numpy 配列で返す。"""
    return np.asarray(MEAN_FRONT_PARTS_LIST, dtype=np.int32)


class affine_convert:
    def __init__(self, afsize, square, temp_parts=None):
        """
        afsize:アフィン変換後の画像サイズ。afsize=[120,120]のように指定
        square:変換後の形状を四角にするか指定。True or False
        temp_parts:テンプレートランドマーク座標 (N, 2) の numpy 配列。None の場合は MEAN_FRONT_PARTS_LIST を使用
        """
        # 顔形状のままアフィン変換を行う際の三角形頂点座標
        triangle_parts_list = [
            [0, 17, 36],
            [0, 1, 36],
            [1, 36, 41],
            [1, 2, 41],
            [2, 40, 41],
            [2, 3, 40],
            [3, 29, 40],
            [3, 29, 31],
            [3, 31, 48],
            [3, 4, 48],
            [4, 5, 48],
            [5, 48, 59],
            [5, 6, 59],
            [6, 58, 59],
            [6, 7, 58],
            [7, 57, 58],
            [7, 8, 57],
            [8, 9, 57],
            [9, 10, 57],
            [10, 56, 57],
            [10, 11, 56],
            [11, 55, 56],
            [11, 12, 55],
            [12, 54, 55],
            [12, 13, 54],
            [13, 35, 54],
            [13, 29, 35],
            [13, 29, 47],
            [13, 14, 47],
            [14, 46, 47],
            [14, 15, 46],
            [15, 45, 46],
            [15, 16, 45],
            [16, 26, 45],
            [17, 18, 36],
            [18, 36, 37],
            [18, 19, 37],
            [19, 37, 38],
            [19, 20, 38],
            [20, 38, 39],
            [20, 21, 39],
            [21, 27, 39],
            [21, 22, 27],
            [22, 27, 42],
            [22, 23, 42],
            [23, 42, 43],
            [23, 24, 43],
            [24, 43, 44],
            [24, 25, 44],
            [25, 44, 45],
            [25, 26, 45],
            [27, 28, 42],
            [27, 28, 39],
            [28, 42, 47],
            [28, 39, 40],
            [28, 29, 40],
            [28, 29, 47],
            [29, 30, 31],
            [29, 30, 35],
            [30, 31, 32],
            [30, 32, 33],
            [30, 33, 34],
            [30, 34, 35],
            [31, 32, 49],
            [31, 48, 49],
            [32, 33, 50],
            [32, 49, 50],
            [33, 34, 52],
            [33, 50, 51],
            [33, 51, 52],
            [34, 35, 53],
            [34, 52, 53],
            [35, 53, 54],
            [36, 37, 41],
            [37, 40, 41],
            [37, 38, 40],
            [38, 39, 40],
            [42, 43, 47],
            [43, 44, 47],
            [44, 46, 47],
            [44, 45, 46],
            [48, 49, 60],
            [48, 59, 60],
            [49, 50, 61],
            [49, 60, 61],
            [50, 51, 61],
            [51, 52, 63],
            [51, 61, 62],
            [51, 62, 63],
            [52, 53, 63],
            [53, 63, 64],
            [53, 54, 64],
            [54, 55, 64],
            [55, 64, 65],
            [55, 56, 65],
            [56, 65, 66],
            [56, 57, 66],
            [57, 58, 66],
            [58, 66, 67],
            [58, 59, 67],
            [59, 60, 67],
            [60, 61, 67],
            [61, 62, 67],
            [62, 66, 67],
            [62, 65, 66],
            [62, 63, 65],
            [63, 64, 65],
            [20, 21, 22],
            [20, 22, 23],
            [19, 20, 24],
            [20, 23, 24],
        ]
        # print(triangle_parts_list)
        # 正方形にアフィン変換を行う際の三角形頂点座標
        triangle_parts_list_square = [
            [0, 1, 36],
            [0, 17, 18],
            [0, 18, 36],
            [1, 31, 41],
            [1, 2, 31],
            [1, 36, 41],
            [2, 3, 48],
            [2, 31, 48],
            [3, 4, 6],
            [3, 6, 48],
            [6, 7, 59],
            [6, 48, 59],
            [7, 8, 58],
            [7, 58, 59],
            [8, 9, 56],
            [8, 56, 57],
            [8, 57, 58],
            [9, 10, 54],
            [9, 54, 55],
            [9, 55, 56],
            [10, 11, 13],
            [10, 13, 54],
            [13, 14, 54],
            [14, 15, 35],
            [14, 35, 54],
            [15, 16, 45],
            [15, 35, 46],
            [15, 45, 46],
            [16, 25, 26],
            [16, 25, 45],
            [18, 19, 37],
            [18, 36, 37],
            [19, 20, 38],
            [19, 37, 38],
            [20, 21, 39],
            [20, 38, 39],
            [21, 22, 27],
            [21, 27, 39],
            [22, 23, 43],
            [22, 27, 42],
            [22, 42, 43],
            [23, 24, 43],
            [24, 25, 44],
            [24, 43, 44],
            [25, 44, 45],
            [27, 28, 39],
            [27, 28, 42],
            [28, 29, 39],
            [28, 29, 42],
            [29, 30, 31],
            [29, 30, 35],
            [29, 31, 40],
            [29, 35, 47],
            [29, 39, 40],
            [29, 42, 47],
            [30, 31, 32],
            [30, 32, 33],
            [30, 33, 34],
            [30, 34, 35],
            [31, 48, 49],
            [31, 49, 50],
            [31, 32, 50],
            [31, 40, 41],
            [32, 33, 50],
            [33, 34, 52],
            [33, 50, 51],
            [33, 51, 52],
            [34, 35, 52],
            [35, 52, 53],
            [35, 53, 54],
            [35, 46, 47],
            [36, 37, 41],
            [37, 38, 41],
            [38, 39, 40],
            [38, 40, 41],
            [42, 43, 47],
            [43, 44, 47],
            [44, 45, 46],
            [44, 46, 47],
            [48, 49, 61],
            [48, 59, 61],
            [49, 50, 61],
            [50, 51, 62],
            [50, 61, 62],
            [51, 52, 62],
            [52, 53, 63],
            [52, 62, 63],
            [53, 54, 63],
            [54, 55, 63],
            [55, 56, 63],
            [56, 57, 62],
            [56, 62, 63],
            [57, 58, 62],
            [58, 59, 61],
            [58, 61, 62],
        ]
        # print(triangle_parts_list_square)
        # 額部の台形特徴点リスト
        forehead_trapezium_parts_list: list = [
            [68, 17, 18],
            [68, 18, 19],
            [68, 69, 19],
            [69, 19, 20],
            [69, 19, 20],
            [69, 20, 21],
            [69, 70, 21],
            [70, 21, 22],
            [70, 71, 22],
            [71, 22, 23],
            [71, 23, 24],
            [71, 72, 24],
            [72, 24, 25],
            [72, 25, 26],
        ]
        if temp_parts is None:
            temp_parts = default_mean_front_parts()
        mean_front_parts_list = np.asarray(temp_parts, dtype=np.float64).copy()

        # 正方形にする使用
        if square:
            mean_front_parts_list = self.shiftTempParts(mean_front_parts_list, afsize)

        self.afsize = afsize
        self.square = square
        self.triangle_parts_list = triangle_parts_list
        self.triangle_parts_list_square = triangle_parts_list_square
        self.mean_front_parts_list = mean_front_parts_list

    # テンプレート特徴点を四角くする
    def shiftTempParts(self, temp_parts, afsize):
        o = 0
        for i, q in enumerate(temp_parts[5:12]):
            temp_parts[i + 5, 0] = o
            o += (np.max(temp_parts[:, 0]) - np.min(temp_parts[:, 0])) / 6
        o = np.max(temp_parts[:, 1]) - np.min(temp_parts[:, 1])
        for i, q in enumerate(temp_parts[0:5]):
            temp_parts[4 - i, 1] = o
            temp_parts[12 + i, 1] = o
            o -= (np.max(temp_parts[:, 1]) - np.min(temp_parts[:, 1])) / 5
        o = 0
        for i, q in enumerate(temp_parts[17:27]):
            temp_parts[i + 17, 0] = o
            o += (np.max(temp_parts[:, 0]) - np.min(temp_parts[:, 0])) / 9

        # 鼻、口、目を下に移動
        temp_parts[27:68, 1] += 30

        # 顔輪郭を四角にしてる
        temp_parts[5:12, 1] = np.max(temp_parts[:, 1])
        temp_parts[0:5, 0] = np.min(temp_parts[:, 0])
        temp_parts[11:17, 0] = np.max(temp_parts[:, 0])
        temp_parts[17:27, 1] = np.min(temp_parts[:, 1])

        # affineの後を正方形にするために計算
        temp_parts[:, 0] = (temp_parts[:, 0] / np.max(temp_parts[:, 0])) * (
            afsize[0] - 1
        )
        temp_parts[:, 1] = (temp_parts[:, 1] / np.max(temp_parts[:, 1])) * (
            afsize[1] - 1
        )

        return temp_parts

    def main(
        self, src, landmark, binary_on=False
    ):  # サーモ温度値をsrcに入力する場合binary_on = True
        # アフィンサイズに応じた0埋め配列生成
        dst = np.zeros((self.afsize[0], self.afsize[1]))

        # RGB三次元化
        dst = np.stack([dst, dst, dst], axis=2)

        # 正方形にアフィン変換
        if self.square:
            if binary_on:
                # サーモ温度値データをRGBの三次元に変換
                src = np.stack([src, src, src], axis=2)
            # 三角形ごとにアフィン変換＆合成
            for traiangles in self.triangle_parts_list_square:
                src_pts = [
                    landmark[traiangles[0]],
                    landmark[traiangles[1]],
                    landmark[traiangles[2]],
                ]
                dst_pts = [
                    self.mean_front_parts_list[traiangles[0]],
                    self.mean_front_parts_list[traiangles[1]],
                    self.mean_front_parts_list[traiangles[2]],
                ]
                self.warp_triangle(src, dst, src_pts, dst_pts)
            if binary_on:
                # サーモ温度値データを1次元に戻す
                dst = cv2.cvtColor(dst.astype(np.float32), cv2.COLOR_BGR2GRAY)

        # 顔形状のままアフィン変換
        else:
            # 三角形ごとにアフィン変換＆合成
            for traiangles in self.triangle_parts_list:
                src_pts = [
                    landmark[traiangles[0]],
                    landmark[traiangles[1]],
                    landmark[traiangles[2]],
                ]
                dst_pts = [
                    self.mean_front_parts_list[traiangles[0]],
                    self.mean_front_parts_list[traiangles[1]],
                    self.mean_front_parts_list[traiangles[2]],
                ]
                self.warp_triangle(src, dst, src_pts, dst_pts)

        return dst

    def warp(self, src, dst, src_pts, dst_pts, transform_func, warp_func, **kwargs):
        src_pts_arr = np.array(src_pts, dtype=np.float32)
        dst_pts_arr = np.array(dst_pts, dtype=np.float32)
        src_rect = cv2.boundingRect(src_pts_arr)
        dst_rect = cv2.boundingRect(dst_pts_arr)
        src_crop = src[
            src_rect[1] : src_rect[1] + src_rect[3],
            src_rect[0] : src_rect[0] + src_rect[2],
        ]
        dst_crop = dst[
            dst_rect[1] : dst_rect[1] + dst_rect[3],
            dst_rect[0] : dst_rect[0] + dst_rect[2],
        ]
        src_pts_crop = src_pts_arr - src_rect[:2]
        dst_pts_crop = dst_pts_arr - dst_rect[:2]
        mat = transform_func(
            src_pts_crop.astype(np.float32), dst_pts_crop.astype(np.float32)
        )
        warp_img = warp_func(
            src_crop,
            mat,
            tuple(dst_rect[2:]),
            borderMode=cv2.BORDER_REPLICATE,
            **kwargs
        )
        mask = np.zeros_like(dst_crop, dtype=np.float32)
        cv2.fillConvexPoly(
            mask, dst_pts_crop.astype(np.int32), (1.0, 1.0, 1.0), cv2.LINE_AA
        )

        dst_crop_merge = warp_img * mask + dst_crop * (1 - mask)
        dst[
            dst_rect[1] : dst_rect[1] + dst_rect[3],
            dst_rect[0] : dst_rect[0] + dst_rect[2],
        ] = dst_crop_merge

    def warp_triangle(self, src, dst, src_pts, dst_pts, **kwargs):
        self.warp(
            src, dst, src_pts, dst_pts, cv2.getAffineTransform, cv2.warpAffine, **kwargs
        )

    def warp_rectangle(self, src, dst, src_pts, dst_pts, **kwargs):
        self.warp(
            src,
            dst,
            src_pts,
            dst_pts,
            cv2.getPerspectiveTransform,
            cv2.warpPerspective,
            **kwargs
        )
