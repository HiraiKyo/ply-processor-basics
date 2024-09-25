import datetime
import random
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray


def detect_plane(
    points: NDArray[np.floating],
    threshold: float = 0.1,
    minPoints: int = 100,
    maxIteration: int = 1000,
) -> Union[Tuple[NDArray[np.floating], NDArray[np.floating]], Tuple[None, None]]:
    """
    点群から最大平面をRANSACで検出する関数

    :param points: a set of points
    :param threshold: threshold for RANSAC
    :param minPoints: minimum number of points for RANSAC
    :param maxIteration: maximum number of iterations for RANSAC
    :return: 検出した平面上の点ポインタ, 平面方程式 ax+by+cz+d=0 の係数(N, 4)

    検出失敗時は None を返す
    """
    plane = Plane()

    plane_model, inliers = plane.fit(points, thresh=threshold, minPoints=minPoints, maxIteration=maxIteration)

    if plane_model == []:
        return None, None

    return inliers, plane_model


class Plane:
    def __init__(self):
        self.inliers = []
        self.equation = []

    def fit(self, pts, thresh=0.1, minPoints=100, maxIteration=1000, normal_samples=10):
        n_points = pts.shape[0]
        best_eq = []
        best_inliers = []

        for it in range(maxIteration):
            random.seed(datetime.datetime.now().microsecond)
            id_samples = random.sample(range(0, n_points), 3)
            pt_samples = pts[id_samples]

            vecA = pt_samples[1, :] - pt_samples[0, :]
            vecB = pt_samples[2, :] - pt_samples[0, :]

            vecC = np.cross(vecA, vecB)
            normC = np.linalg.norm(vecC)

            # ランダムサンプリングした3点がほぼ同一直線上にある場合はスキップ
            if normC < 1e-6:
                continue

            vecC = vecC / normC
            k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
            plane_eq = [vecC[0], vecC[1], vecC[2], k]

            pt_id_inliers = []
            dist_pt = (
                plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
            ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

            pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
            if len(pt_id_inliers) > len(best_inliers):
                best_eq = plane_eq
                best_inliers = pt_id_inliers

        self.inliers = best_inliers
        self.equation = best_eq

        if len(self.inliers) < minPoints:
            return [], []

        return self.equation, self.inliers
