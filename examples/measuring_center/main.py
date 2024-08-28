import open3d as o3d
import numpy as np
from ply_processor_basics.points.convex_hull import detect_line
from ply_processor_basics.points.ransac import detect_plane
from ply_processor_basics.points.clustering import plane_clustering
from ply_processor_basics.points.convex_hull.detect_circle import fit_circle
def visualize(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def main():
    pcd = o3d.io.read_point_cloud("sample.ply")
    # o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")
    points = np.asarray(pcd.points)

    # エッジ検出
    plane_indices, plane_model = detect_plane(points, threshold=1.0)
    indices, line_segments_indices, line_models = detect_line(points[plane_indices], plane_model)

    # エッジを可視化
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points[plane_indices])
    line_set.lines = o3d.utility.Vector2iVector(line_segments_indices)
    # o3d.visualization.draw_geometries([line_set], window_name="Detected Edge")

    # 円筒検出
    # 円筒上面の点群を取得
    loops = 5
    tmp = points
    for i in range(loops):
        outlier_indices = np.setdiff1d(np.arange(len(tmp)), plane_indices)
        tmp = tmp[outlier_indices]
        plane_indices, plane_model = detect_plane(tmp, threshold=1.0)

    # クラスタリングを行う
    clusters = plane_clustering(tmp[plane_indices], eps = 10.0)
    if len(clusters) == 0:
        return
    points_cylinder_top = tmp[plane_indices][clusters[0]]
    print(points_cylinder_top.shape)
    # 円盤フィッティング
    center, normal, radius = fit_circle(points_cylinder_top)

    # 円盤とエッジを描画
    mesh_disk = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=0.1, resolution=50, split=1)
    mesh_disk.compute_vertex_normals()
    mesh_disk.translate(center)
    mesh_disk.rotate(mesh_disk.get_rotation_matrix_from_xyz((np.pi/2, 0, 0)), center)
    o3d.visualization.draw_geometries([line_set, pcd, mesh_disk])

if __name__ == "__main__":
    main()

