# ply-processor-basics

Basic libraries for manipulating point cloud.

## Installation

```sh
pip install git+https://github.com/HiraiKyo/ply-processor-basics
```

## Methods

### STL

#### `stl.stl2ply`

### Vector

#### `vector.normalize`

### Matrix

#### `matrix.get_rotation_from_vectors`

### Points

#### `points.transform_to_plane_coordinates`

#### `points.get_distances_to_line`

#### `points.get_distances_to_plane`

#### `points.rotate_euler`

#### `points.clip_by_plane`

#### `points.ransac.detect_plane`

### Open3d

#### `pcd.snapshot`

## サンプルデータ寸法

- 円柱中心: (49.0, 52.0)
- 円柱半径: 17.5
- 円柱高さ: 40.0
- エッジ面: 122.0
