from typing import Callable, List, Optional

import numpy as np
import open3d as o3d
from numpy.typing import NDArray


def snapshot(
    pcds: List[o3d.geometry.PointCloud],
    save_path: str,
    cam_front: NDArray[np.floating],
    cam_lookat: NDArray[np.floating],
    cam_up: NDArray[np.floating],
    cam_zoom: float,
    on_finish: Optional[Callable],
) -> None:
    """
    Asynchronously save point cloud as a snapshot.

    :param pcd: Point cloud to save.
    :param save_path: Path to save.
    :param cam_front: Front vector of the camera.
    :param cam_lookat: Lookat vector of the camera.
    :param cam_up: Up vector of the camera.
    :param cam_zoom: Zoom of the camera.
    """

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    for pcd in pcds:
        vis.add_geometry(pcd)
    print(vis)
    opt = vis.get_render_option()
    if opt is None:
        print("[ply_processor_basics] Open3d Warning: No render option")
    else:
        opt.show_coordinate_frame = True
        opt.mesh_show_back_face = True
        opt.mesh_show_wireframe = True
        opt.background_color = np.asarray([1, 1, 1])

    ctr = vis.get_view_control()
    if ctr is None:
        print("[ply_processor_basics] Open3d Warning: No view control")
    else:
        ctr.set_zoom(cam_zoom)
        ctr.set_front(cam_front)
        ctr.set_lookat(cam_lookat)
        ctr.set_up(cam_up)

    for pcd in pcds:
        vis.update_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path, do_render=True)

    vis.destroy_window()

    if on_finish:
        on_finish()
