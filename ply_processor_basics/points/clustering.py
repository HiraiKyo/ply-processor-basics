import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN


def plane_clustering(points: NDArray[np.floating], eps: float = 1.0, min_samples: int = 100):
    """
    DBSCANによる平面上の点群のクラスタリングを行う

    :param points: 平面上の点群(N, 2)
    :return: クラスタ点数の多い順にソートされたクラスタ点群ポインタ(N, M)
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    clusters = dbscan.fit_predict(points[:, :2])
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    # クラスタリングできなかった点群(-1)は除外
    unique_clusters = unique_clusters[unique_clusters != -1]
    # 各クラスタの点群ポインタを返す
    cluster_indices = [np.where(clusters == cluster)[0] for cluster in unique_clusters]
    return cluster_indices
