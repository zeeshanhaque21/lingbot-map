import json

import numpy as np
import open3d as o3d
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from lingbot_map.reconstruction.io import digest
from lingbot_map.reconstruction.panorama import (
    attach_refinement,
    stitch_sweep,
    sweep_rotations,
)
from lingbot_map.reconstruction.rendering import SurfaceRenderer
from lingbot_map.reconstruction.tour import panorama_rays


def test_spherical_render_pose_depth_and_wrap():
    # Distinct colors on all six walls expose flipped axes and seams.
    vertices, faces, colors = [], [], []
    palette = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    )
    for axis in range(3):
        for sign in (-1, 1):
            points = np.zeros((4, 3))
            points[:, axis] = sign * 2
            points[:, [i for i in range(3) if i != axis]] = [
                [-2, -2],
                [2, -2],
                [2, 2],
                [-2, 2],
            ]
            start = len(vertices)
            vertices.extend(points)
            faces.extend([[start, start + 1, start + 2], [start, start + 2, start + 3]])
            colors.extend([palette[len(colors) // 4]] * 4)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [20, 35, -10], degrees=True).as_matrix()
    pose[:3, 3] = [3, -2, 5]
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices) @ pose[:3, :3].T + pose[:3, 3]),
        o3d.utility.Vector3iVector(faces),
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    rays = panorama_rays(pose, 256, 128)
    rgb, depth, visible = SurfaceRenderer(mesh).render_rays(rays)
    assert visible.all()
    camera_rays = rays[..., 3:] @ pose[:3, :3]
    expected_depth = 2 / np.abs(camera_rays).max(axis=-1)
    np.testing.assert_allclose(depth, expected_depth, atol=1e-5)
    for v, u, expected in [
        (64, 128, 5),
        (64, 192, 1),
        (64, 64, 0),
        (0, 128, 2),
        (127, 128, 3),
        (64, 0, 4),
    ]:
        np.testing.assert_array_equal(rgb[v, u], palette[expected] * 255)
    np.testing.assert_array_equal(rgb[:, 0], rgb[:, -1])
    with pytest.raises(ValueError):
        panorama_rays(np.diag([-1, 1, 1, 1]), 256, 128)


def test_sweep_stitch_covers_sphere_and_preserves_direction(tmp_path):
    width, height = 208, 120
    k = np.array([[171, 0, 103.5], [0, 174, 59.5], [0, 0, 1]])
    rotations = sweep_rotations()
    np.testing.assert_allclose(rotations[0], rotations[-1])
    image_root = tmp_path / "images"
    image_root.mkdir()
    frames = []
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(k).T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    for index, rotation in enumerate(rotations):
        directions = rays @ rotation.T
        Image.fromarray(((directions + 1) * 127.5).round().astype(np.uint8)).save(
            image_root / f"{index:03d}.png"
        )
        frames.append({"rotation": rotation.tolist()})
    (tmp_path / "sweep.json").write_text(
        json.dumps(
            {
                "width": width,
                "height": height,
                "intrinsics": k.tolist(),
                "frames": frames,
            }
        )
    )
    result = stitch_sweep(tmp_path, image_root, tmp_path / "stitch", width=256)
    assert result["camera_coverage_fraction"] > 0.999
    actual = np.asarray(Image.open(tmp_path / "stitch/panorama.png"), dtype=float) / 255
    expected = (panorama_rays(np.eye(4), 256, 128)[..., 3:] + 1) / 2
    assert np.abs(actual - expected).mean() < 0.002
    assert result["wrap_seam_rgb_difference"] < 0.01


def test_attach_keeps_raw_and_rejects_wrong_station(tmp_path):
    refinement, sweep = tmp_path / "refinement", tmp_path / "sweep"
    refinement.mkdir()
    sweep.mkdir()
    original = [{"id": "000", "panorama": "raw.png", "generated": False}]
    (tmp_path / "stations.json").write_text(json.dumps(original))
    # Explicit synthetic fixture, never a model output or production tour.
    Image.new("RGB", (64, 32), "red").save(refinement / "panorama.png")
    (refinement / "stitch.json").write_text(
        json.dumps(
            {
                "generated": True,
                "camera_coverage_fraction": 1.0,
                "panorama_sha256": digest(refinement / "panorama.png"),
                "source_sweep": str(sweep),
            }
        )
    )
    (sweep / "sweep.json").write_text(
        json.dumps({"tour": str(tmp_path), "station": "001"})
    )
    with pytest.raises(ValueError, match="different tour or station"):
        attach_refinement(tmp_path, "000", refinement)
    assert json.loads((tmp_path / "stations.json").read_text()) == original
    (sweep / "sweep.json").write_text(
        json.dumps({"tour": str(tmp_path), "station": "000"})
    )
    station = attach_refinement(tmp_path, "000", refinement)
    assert station["panorama"] == "raw.png"
    assert station["generated"] is False
    assert station["refinement_review"] == "unverified_generated_candidate"
    with pytest.raises(ValueError, match="already attached"):
        attach_refinement(tmp_path, "000", refinement)
