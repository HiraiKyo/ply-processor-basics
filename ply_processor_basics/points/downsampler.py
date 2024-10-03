import numpy as np
import open3d as o3d
from numpy.typing import NDArray


def voxel_grid_filter(points: NDArray[np.floating], voxel_size: float) -> NDArray[np.floating]:
    """
    VoxelGridフィルタリング

    :param points: 点群(N, 3)
    :param voxel_size: ボクセルサイズ
    Returns:
        NDArray: フィルタリング後の点群
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd.points)
