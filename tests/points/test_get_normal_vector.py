import datetime

import numpy as np
import pytest

from ply_processor_basics.points import get_normal_vector

# テスト用法線ベクトルをランダム生成
np.random.seed(datetime.datetime.now().microsecond)
test_normal = np.random.rand(3)
test_normal = test_normal / np.linalg.norm(test_normal)
# test_normalを法線とする平面上にランダム点群を生成
if np.allclose(test_normal, [1, 0, 0]):
    u = np.array([0, 1, 0])
else:
    u = np.cross(test_normal, [1, 0, 0])
u = u / np.linalg.norm(u)
v = np.cross(test_normal, u)

random_coords = np.random.rand(100, 2) * 2 - 1
test_points = 10 * (random_coords[:, 0][:, np.newaxis] * u + random_coords[:, 1][:, np.newaxis] * v)


@pytest.mark.parametrize("points", [test_points])
@pytest.mark.parametrize("normal", [test_normal])
def test_simple(normal, points):
    normal_vector = get_normal_vector(points)
    # 法線ベクトルの正規化
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    # 法線ベクトルの一致を確認(法線ベクトルは2種類)
    assert np.allclose(normal_vector, normal, 1e-1) or np.allclose(normal_vector, -normal, 1e-1)
