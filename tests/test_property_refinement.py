import json
from pathlib import Path

import numpy as np

from lingbot_map.reconstruction.io import write_json,write_npz
from lingbot_map.reconstruction.refinement import refine


def test_colmap_refinement_calibrates_depth_and_preserves_global_cameras(tmp_path):
    source=tmp_path/"source";model=tmp_path/"colmap";destination=tmp_path/"refined"
    (source/"frames").mkdir(parents=True);model.mkdir()
    write_json(source/"input.json",{"frames":[{"file":f"frames/{i:06d}.jpg"} for i in range(8)]})
    write_json(source/"inference.json",{"window":8,"overlap":2})
    e=np.repeat(np.eye(4,dtype=np.float32)[None,:3],8,0)
    k=np.repeat(np.array([[10.,0,3],[0,10.,3],[0,0,1.]])[None],8,0)
    write_npz(source/"windows/000000.npz",frame_ids=np.arange(8),depth=np.ones((8,7,7),np.float32),
              intrinsics=k,extrinsics=e,confidence=np.full((8,7,7),10.,np.float32),rgb=np.zeros((8,7,7,3),np.uint8))
    (model/"cameras.txt").write_text("# camera\n1 PINHOLE 7 7 10 10 3 3\n")
    point_rows=[];observations=[]
    for y in range(7):
        for x in range(7):
            identifier=1+y*7+x
            point_rows.append(f"{identifier} {(x-3)*.2} {(y-3)*.2} 2 100 100 100 0.1 1 0 2 0 3 0")
            observations.append(f"{x} {y} {identifier}")
    (model/"points3D.txt").write_text("\n".join(point_rows)+"\n")
    (model/"images.txt").write_text("".join(f"{i+1} 1 0 0 0 0 0 0 1 {i:06d}.jpg\n"+" ".join(observations)+"\n" for i in range(8)))
    refine(source,destination,model)
    with np.load(destination/"windows/000000.npz") as result:
        assert result["registered"].all()
        np.testing.assert_allclose(result["depth"],2)
        np.testing.assert_allclose(result["intrinsics"],k)
        np.testing.assert_allclose(result["extrinsics"],e)
    assert json.loads((destination/"inference.json").read_text())["poses_global"]
    assert not json.loads((destination/"depth-calibration.json").read_text())["metric_scale_known"]
