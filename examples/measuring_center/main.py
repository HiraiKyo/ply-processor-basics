from pprint import pprint
import open3d as o3d
import numpy as np
from ply_processor_basics.points.convex_hull import detect_line
from ply_processor_basics.points.ransac import detect_plane
from ply_processor_basics.points.clustering import plane_clustering
from ply_processor_basics.points.convex_hull import detect_circle

def visualize(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def line_distance(p0, v0, p1, v1):
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)

    w0 = p0 - p1
    a = np.cross(v0, v1)
    b = np.linalg.norm(a)

    if b == 0:
        return np.linalg.norm(np.cross(w0, v0))

    return abs(np.dot(w0, a)) / b

def main():
    pcd = o3d.io.read_point_cloud("sample.ply")
    points = np.asarray(pcd.points)
    # visualize(points)

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
    # visualize(points_cylinder_top)

    # 円盤フィッティング
    indices, center, normal, radius = detect_circle(points_cylinder_top, plane_model)

    # 円盤とエッジを描画
    mesh_disk = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=3.0, resolution=50, split=1)
    mesh_disk.compute_vertex_normals()
    normal = normal / np.linalg.norm(normal)
    rotation_axis = np.cross(np.array([0, 0, 1]), normal)
    rotation_angle = np.arccos(np.dot(np.array([0, 0, 1]), normal))
    rotation_matrix = mesh_disk.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
    mesh_disk.rotate(rotation_matrix, center=(0, 0, 0))
    mesh_disk.translate(center)
    o3d.visualization.draw_geometries([line_set, pcd, mesh_disk])

    # 各エッジと円筒軸の間の距離を算出
    distances = []
    for line_model in line_models:
        p, v = line_model
        distances.append(line_distance(center, normal, p, v))
    print("円筒中心: ", center)
    print("円筒半径: ", radius)
    # 配列をpretty print
    print("エッジと円筒軸の距離: ")
    pprint([f"{d:.2f}" for d in distances])

if __name__ == "__main__":
    main()

