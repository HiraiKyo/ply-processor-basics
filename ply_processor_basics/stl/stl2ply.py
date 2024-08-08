#!/usr/bin/env /usr/bin/python3

from typing import List

import numpy as np
import open3d as o3d


def stl2ply(
    name: str, cam_dir: List[float] = [-100.0, -100, 100], sample_points: int = 1000000, voxel_size: float = 1.0
):
    mesh = o3d.io.read_triangle_mesh(name + ".stl")
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
    pcdd = pcd.voxel_down_sample(voxel_size=voxel_size)

    diameter = np.linalg.norm(np.asarray(pcdd.get_max_bound()) - np.asarray(pcdd.get_min_bound()))
    camera = np.array(cam_dir)
    radius = diameter * 100

    _, pt_map = pcdd.hidden_point_removal(camera, radius)

    pcd = pcdd.select_by_index(pt_map)

    o3d.io.write_point_cloud(name + ".ply", pcd)
