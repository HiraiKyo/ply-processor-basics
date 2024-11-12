from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d
import pytest
from numpy.typing import NDArray

from ply_processor_basics.points.ml import match


@pytest.fixture(scope="session")
def sample_data() -> Tuple[NDArray, NDArray]:
    """
    ダウンサンプリングされたテスト用の点群データを読み込む
    """

    def load_and_downsample_point_cloud(file_path: str, voxel_size: float = 0.05) -> NDArray:
        # 点群の読み込み
        pcd = o3d.io.read_point_cloud(file_path)

        # ダウンサンプリング
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

        points = np.asarray(pcd_down.points)
        return points

    # テストデータのパス
    model_path = Path("data/samples/ml/model.ply")
    scan_path = Path("data/samples/ml/scn.ply")

    assert model_path.exists(), f"Model file not found: {model_path}"
    assert scan_path.exists(), f"Scan file not found: {scan_path}"

    # ダウンサンプリングしたデータを読み込む
    model_points = load_and_downsample_point_cloud(str(model_path))
    scan_points = load_and_downsample_point_cloud(str(scan_path))

    return model_points, scan_points


def test_match_basic_functionality(sample_data: Tuple[NDArray, NDArray]):
    """
    基本的な機能テスト
    """
    model_points, scan_points = sample_data

    # マッチング実行
    matches = match(model_points, scan_points)

    # 基本的な検証
    assert isinstance(matches, np.ndarray), "戻り値はnumpy配列であること"
    assert matches.dtype == np.int64, "戻り値は整数型であること"
    assert len(matches) == len(scan_points), "マッチングインデックスはターゲット点群と同じ長さであること"
    assert np.all(matches >= 0) and np.all(matches < len(model_points)), "すべてのインデックスが有効な範囲内であること"
