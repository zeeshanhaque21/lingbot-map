"""Exercise the complete textured GLB validator beyond OpenCV's remap row limit."""

import json

import numpy as np
import trimesh
from PIL import Image

from lingbot_map.reconstruction.validation import validate


def test_textured_plane_with_more_than_32767_visible_pixels(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    windows = tmp_path / "windows"
    windows.mkdir()
    rgb = np.full((256, 256, 3), [64, 128, 192], dtype=np.uint8)
    mesh = trimesh.Trimesh(
        vertices=[[-4, -4, 2], [4, -4, 2], [4, 4, 2], [-4, 4, 2]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=[[0, 1], [1, 1], [1, 0], [0, 0]],
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(rgb[:4, :4]), metallicFactor=0
        ),
    )
    mesh.apply_transform(np.diag([1.0, -1.0, -1.0, 1.0]))
    mesh.export(root / "property.glb")
    camera = {
        "frame": 0,
        "window": 0,
        "camera_to_world": np.eye(4).tolist(),
        "intrinsics": [[64, 0, 128], [0, 64, 128], [0, 0, 1]],
        "held_out_from_fusion": True,
        "timestamp_seconds": 0,
    }
    (root / "cameras.json").write_text(json.dumps([camera]))
    (root / "alignment.json").write_text(
        json.dumps({"transforms": [np.eye(4).tolist()]})
    )
    np.savez_compressed(
        windows / "000000.npz",
        frame_ids=[0],
        rgb=rgb[None],
        depth=np.full((1, 256, 256), 2, dtype=np.float32),
        confidence=np.ones((1, 256, 256), dtype=np.float32),
    )
    report = validate(tmp_path, asset_name="property.glb")
    assert report["median_rendered_coverage"] == 1
    assert report["median_supported_depth_fraction"] == 1
    assert report["median_absolute_rgb_error_visible"] == 0
    assert report["view_consistency_gate"]
