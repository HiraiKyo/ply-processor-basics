import numpy as np
from numpy.typing import NDArray


def get_normal_vector(points: NDArray[np.floating]) -> NDArray[np.floating]:
    """_summary_
    点群データに対して主成分分析を行い、最小の固有値の固有ベクトルを返す（法線ベクトル）

    :param points: 点群データ(N, 3)
    :returns: 法線ベクトル
    """
    # 重心
    centeroid = np.mean(points, axis=0)
    centered_points = points - centeroid

    # 共分散行列
    cov_m = np.cov(centered_points.T)

    # 固有値と固有ベクトルを計算
    eigenvalues, eigenvectors = np.linalg.eig(cov_m)

    # 最小の固有値に対応する固有ベクトルを返す
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    return normal_vector.real
