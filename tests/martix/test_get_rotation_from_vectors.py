import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from ply_processor_basics.matrix import get_rotation_from_vectors


def test_identity_rotation():
    """同一ベクトルに対して単位行列が返されることをテスト"""
    vec = np.array([1, 0, 0])
    result = get_rotation_from_vectors(vec, vec)
    assert_array_almost_equal(result, np.eye(3))


def test_90_degree_rotation():
    """90度回転のテスト"""
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    result = get_rotation_from_vectors(vec1, vec2)
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert_array_almost_equal(result, expected)


def test_180_degree_rotation():
    """180度回転のテスト"""
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([-1, 0, 0])
    result = get_rotation_from_vectors(vec1, vec2)
    expected = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    assert_array_almost_equal(result, expected)


def test_arbitrary_rotation():
    """任意の回転のテスト"""
    vec1 = np.array([1, 1, 1])
    vec2 = np.array([1, 2, 3])
    result = get_rotation_from_vectors(vec1, vec2)
    # 結果の検証：回転後のvec1がvec2と同じ方向を向いているか
    rotated_vec1 = np.dot(result, vec1)
    assert_array_almost_equal(rotated_vec1 / np.linalg.norm(rotated_vec1), vec2 / np.linalg.norm(vec2))


@pytest.mark.parametrize(
    "vec1, vec2",
    [
        (np.array([1, 0, 0]), np.array([0, 0, 1])),
        (np.array([1, 1, 1]), np.array([-1, -1, -1])),
        (np.array([1, 0, 0]), np.array([0.707, 0.707, 0])),
        (np.array([1, 2, 3]), np.array([3, 2, 1])),
    ],
)
def test_various_rotations(vec1, vec2):
    """さまざまな回転のテスト"""
    result = get_rotation_from_vectors(vec1, vec2)
    rotated_vec1 = np.dot(result, vec1)
    assert_array_almost_equal(rotated_vec1 / np.linalg.norm(rotated_vec1), vec2 / np.linalg.norm(vec2))


def test_small_angle_rotation():
    """小さな角度の回転のテスト"""
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0.9999, 0.0141, 0])  # 約0.81度の回転
    result = get_rotation_from_vectors(vec1, vec2)
    rotated_vec1 = np.dot(result, vec1)
    assert_array_almost_equal(rotated_vec1, vec2, decimal=4)


def test_rotation_properties():
    """回転行列の性質をテスト"""
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([3, 2, 1])
    R = get_rotation_from_vectors(vec1, vec2)
