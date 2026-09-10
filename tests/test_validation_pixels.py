import json

import numpy as np
import open3d as o3d

from lingbot_map.reconstruction.validation import validate


def test_validation_samples_integer_centered_captured_pixels(tmp_path):
    model, windows = tmp_path / "model", tmp_path / "windows"
    model.mkdir()
    windows.mkdir()
    k = np.array([[32.0, 0, 32], [0, 32, 32], [0, 0, 1]])
    uv = np.array([[-1.0, -1], [64, -1], [64, 64], [-1, 64]])
    vertices = np.column_stack([(uv - 32) / 16, np.full(4, 2.0)])
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]),
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.column_stack([(uv + 1) / 65, np.zeros(4)])
    )
    o3d.io.write_triangle_mesh(str(model / "observed-surfaces.ply"), mesh)
    y, x = np.indices((64, 64))
    rgb = np.rint(np.stack([x + 1, y + 1, np.zeros_like(x)], -1) / 65 * 255).astype(
        "uint8"
    )
    np.savez(
        windows / "000000.npz",
        frame_ids=[0],
        rgb=rgb[None],
        depth=np.full((1, 64, 64), 2.0),
        confidence=np.ones((1, 64, 64)),
    )
    (model / "cameras.json").write_text(
        json.dumps(
            [
                {
                    "frame": 0,
                    "window": 0,
                    "camera_to_world": np.eye(4).tolist(),
                    "intrinsics": k.tolist(),
                    "held_out_from_fusion": True,
                    "timestamp_seconds": None,
                    "pixel_center_offset": 0.0,
                }
            ]
        )
    )
    (model / "alignment.json").write_text(
        json.dumps({"transforms": [np.eye(4).tolist()]})
    )
    result = validate(tmp_path)
    assert result["median_rendered_coverage"] == 1
    assert result["median_absolute_rgb_error_visible"] < 0.001
