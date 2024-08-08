import os

from ply_processor_basics.stl import stl2ply


def test_stl2ply_simple():
    filename = "data/samples/stl2ply"
    stl2ply(filename)
    # ファイルがdata/test.plyに生成されている事を確認
    assert os.path.exists(f"{filename}.ply")
