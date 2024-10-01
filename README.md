# ply-processor-basics

Basic libraries for manipulating point cloud.

## Installation

```sh
pip install git+https://github.com/HiraiKyo/ply-processor-basics@v0.2.0#egg=ply_processor_basics
```

## Methods

### STL

#### `stl.stl2ply`

### Vector

#### `vector.normalize`

#### `vector.estimate_vector`

ベクトルの集合から四分位範囲で平均ベクトルを算出する

#### `vector.ensure_consistent_direction`

基準ベクトル方向にベクトル群を反転する（法線ベクトルが2種類出る対応に用いる）

### Matrix

#### `matrix.get_rotation_from_vectors`

### Points

#### `points.transform_to_plane_coordinates`

#### `points.get_distances_to_line`

#### `points.get_distances_to_plane`

#### `points.get_normal_vector`

点群から法線ベクトルを算出する。

#### `points.rotate_euler`

#### `points.clip_by_plane`

#### `points.plane_clustering`

#### `points.ransac.detect_plane`

#### `points.ransac.detect_circle`

WARN: Deprecated

`points.convex_hull.detect_circle`を使ってください。

#### `points.ransac.detect_line`

#### `points.convex_hull.detect_circle`

#### `points.convex_hull.detect_plane`

#### `points.convex_hull.detect_line`

updated 0.5.0: changed return type

### Open3d

#### `pcd.snapshot`

## Development

### Setup

1. Install `poetry`

2. Install dependencies

```sh
poetry install
```

### Running test

`visual`タグはopen3d.geometry等で表示を確認する用なので、テスト実行時は外す

```sh
poetry run pytest -s {filepath} -m "not visual"
```

TDD開発時にopen3dで表示を確認しつつ進める場合には、そのテストに`@pytest.mark.visual`タグを付けて自動テストに影響しないようにする。

## サンプルデータ寸法

- 円柱中心: (49.0, 52.0)
- 円柱半径: 17.5
- 円柱高さ: 40.0
- エッジ面: 122.0

## Development

### Running test

`visual`タグはopen3d.geometry等で表示を確認する用なので、テスト実行時は外す

```sh
poetry run pytest -s {filepath} -m "not visual"
```

TDD開発時にopen3dで表示を確認しつつ進める場合には、そのテストに`@pytest.mark.visual`タグを付けて自動テストに影響しないようにする。

### Merge Request Checklist

- [] Pytest written and passed.
- [] Types written.
- [] Docstring written.
- [] `README.md` updated if added new API.
- [] `__init__.py` updated if added new API.
