from typing import List, Tuple

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from ply_processor_basics.points import get_distances_to_line, voxel_grid_filter
from ply_processor_basics.points.ransac import detect_line

from .detect_plane_edge import detect_plane_edge


def detect_edge_as_line(
    points: NDArray[np.floating],
    plane_model: NDArray[np.floating],
    threshold: float = 0.5,
    expected_edges: int = 4,
    edge_density: int = 5,
    downsample_voxel_size: float = 5.0,
) -> List[Tuple[NDArray[np.intp], NDArray[np.floating], NDArray[np.floating]]]:
    """
    ConvexHullを用いて平面の外形直線検出

    :param points: 点群(N, 3)
    :param plane_model: 平面モデル(4,)
    :param expected_edges: 期待されるエッジ数
    :param edge_density: エッジ点の密度(詳細はREADME.md参照)
    Returns:
        List[Tuple]: 各エッジに関する以下の情報を含むタプルのリスト
            1. エッジ点のポインタ: shape (N,) の numpy 配列
            2. エッジの線分: shape (2, 3) の numpy 配列
            3. エッジの直線方程式 p+tv=0: shape (2, 3) の numpy 配列
    """
    return_list: List[Tuple[NDArray[np.intp], NDArray[np.floating], NDArray[np.floating]]] = []

    # エッジ点を抽出
    edge_inliers = np.array([], dtype=np.intp)
    for _ in range(edge_density):
        outliers = np.where(np.logical_not(np.isin(np.arange(len(points)), edge_inliers)))[0]
        inliers, _ = detect_plane_edge(points[outliers], plane_model)
        edge_inliers = np.concatenate([edge_inliers, outliers[inliers]])

    # ConvexHullによる角過密の解消
    o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[edge_inliers]))])
    downsampled_points = voxel_grid_filter(points[edge_inliers], voxel_size=downsample_voxel_size)
    downsampled_inliers = np.arange(len(downsampled_points))
    o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(downsampled_points))])
    # 抽出点から直線を検出
    for i in range(expected_edges):
        tmp_inliers, line_model = detect_line(downsampled_points[downsampled_inliers], threshold=threshold)
        inliers = downsampled_inliers[tmp_inliers]

        # 線分検出失敗時はこれまでの計算結果のみを返す
        if line_model is None:
            return return_list

        # ノイズ対応、直線上の線分端点を取得
        # FIXME: 点数が少なすぎてDBSCANクラスタリングが使えない
        # clusters = line_clustering(points[inliers], min_samples=1)
        # if len(clusters) == 0:
        #     continue
        # cluster = clusters[0]
        cluster = inliers

        p, v = line_model
        projected_points = np.dot(points[cluster] - p, v)  # 点pから直線上の点への射影距離
        sorted_indices = np.argsort(projected_points)
        sorted_points = points[cluster[sorted_indices]]
        start_point: NDArray[np.floating] = sorted_points[0]
        end_point: NDArray[np.floating] = sorted_points[-1]

        # 検出した線分をエッジ点から消去
        downsampled_inliers = downsampled_inliers[np.where(np.logical_not(np.isin(downsampled_inliers, cluster)))[0]]

        # 元のエッジ点群の直線近傍点を抽出
        distances = get_distances_to_line(points[edge_inliers], line_model[0], line_model[1])
        inliers = np.where(distances < threshold)[0]
        # CAUTION: クラスタリング必要かも

        return_list.append((edge_inliers[inliers], np.array([start_point, end_point]), line_model))

    return return_list
