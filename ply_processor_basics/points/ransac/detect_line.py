import datetime
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ply_processor_basics.points import get_distances_to_line


def detect_line(
    points: NDArray[np.floating],
    threshold: float = 0.1,
    max_iteration: int = 1000,
    recursive: bool = True,
) -> Tuple[NDArray[np.intp], Union[NDArray[np.floating], None]]:
    """
    点群データから最大点数の直線を検出する関数

    :param points: 点群データ(N, 3)
    :threshold: 抽出距離閾値
    :max_iteration: 最大繰り返し回数
    :recursive: 再帰的に直線を検出するかどうか
    :return: 直線の点群ポインタ(N, ), 直線の方程式p+tv=0
    """
    best_inliers = np.array([], dtype=np.intp)
    best_model = None

    for _ in range(max_iteration):
        np.random.seed(datetime.datetime.now().microsecond)
        sample_indices = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[sample_indices]

        # 直線の方向ベクトルを計算
        v = p2 - p1
        v = v / np.linalg.norm(v)
        p = p1

        distances = get_distances_to_line(points, p, v)
        inliers = np.where(distances < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = np.asarray([p, v])

    if recursive:
        inliers, best_model = detect_line(points[best_inliers], threshold / 2, max_iteration, recursive=False)
        best_inliers = best_inliers[inliers]
    return best_inliers, best_model
