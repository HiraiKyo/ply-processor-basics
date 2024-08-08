import os

import numpy as np
import open3d as o3d

from ply_processor_basics.pcd import snapshot


def test_simple():
    pcd = o3d.io.read_point_cloud("data/samples/sample.ply")
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    filepath = "data/samples/sample.png"
    cam_front = np.array([-100, -100, 100])
    cam_lookat = np.array([0, 0, 0])
    cam_up = np.array([0, 0, 1])
    cam_zoom = 0.2

    def on_finish_callback():
        print("done")

    snapshot([pcd], filepath, cam_front, cam_lookat, cam_up, cam_zoom, on_finish_callback)
    # ファイルの存在確認
    assert os.path.exists(filepath)

    # ファイルの削除
    os.remove(filepath)
