import open3d as o3d
import numpy as np
from ply_processor_basics.points.convex_hull import detect_plane_edge
from ply_processor_basics.points.ransac import detect_plane


def main():
    pcd = o3d.io.read_point_cloud("sample.ply")
    o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")
    points = np.asarray(pcd.points)

    # エッジ検出
    plane_indices, plane_model = detect_plane(points, threshold=1.0)
    edge_indices = detect_plane_edge(points[plane_indices], plane_model)

    # エッジを可視化
    pcd_edge = o3d.geometry.PointCloud()
    pcd_edge.points = o3d.utility.Vector3dVector(points[plane_indices][edge_indices])
    o3d.visualization.draw_geometries([pcd_edge], window_name="Detected Edge")

if __name__ == "__main__":
    main()