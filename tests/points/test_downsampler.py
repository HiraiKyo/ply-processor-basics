import datetime

import numpy as np
import open3d as o3d
import pytest

from ply_processor_basics.points import voxel_grid_filter

np.random.seed(datetime.datetime.now().microsecond)
# 10x10x10の立方体内にランダムに点を100000個生成
test_points = np.random.rand(1000000, 3) * 10


@pytest.mark.parametrize("points", [test_points])
def test_success(points):
    downsampled_points = voxel_grid_filter(points, 1.0)


@pytest.mark.parametrize("points", [test_points])
@pytest.mark.visual
def test_visual(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

    downsampled_points = voxel_grid_filter(points, 1.0)
    pcd.points = o3d.utility.Vector3dVector(downsampled_points)
    o3d.visualization.draw_geometries([pcd])
