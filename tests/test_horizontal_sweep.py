import json

import numpy as np
import open3d as o3d
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from lingbot_map.reconstruction.panorama import export_sweep, horizontal_rotations


def test_horizontal_rotation_preserves_elevation_for_tilted_camera():
    source = Rotation.from_euler("xyz", [-20, 35, 7], degrees=True).as_matrix()
    world_up = np.array([0.0, -1, 0])
    rotations = horizontal_rotations(source, world_up, frame_count=121)
    poses = source @ np.asarray(rotations)
    np.testing.assert_allclose(
        poses[:, :, 2] @ world_up, source[:, 2] @ world_up, atol=1e-12
    )
    np.testing.assert_allclose(
        poses[:, :, 1] @ world_up, source[:, 1] @ world_up, atol=1e-12
    )
    np.testing.assert_allclose(
        Rotation.from_matrix(poses[:-1].transpose(0, 2, 1) @ poses[1:]).magnitude(),
        np.deg2rad(3),
        atol=1e-12,
    )
    np.testing.assert_array_equal(rotations[0], rotations[-1])
    with pytest.raises(ValueError):
        horizontal_rotations(source, None)
    with pytest.raises(ValueError):
        horizontal_rotations(source, [0, 0, 0])


def test_horizontal_export_faces_four_walls_and_keeps_photo_anchors(tmp_path):
    # Four differently colored walls expose an accidental pitch or axis flip.
    vertices, faces, colors = [], [], []
    palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]
    for axis, sign, color in [
        (2, 1, palette[0]),
        (0, 1, palette[1]),
        (2, -1, palette[2]),
        (0, -1, palette[3]),
    ]:
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
        colors.extend([np.array(color) / 255] * 4)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces)
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh(str(tmp_path / "render-mesh.ply"), mesh)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("x", -20, degrees=True).as_matrix()
    k = [[60.0, 0, 31.5], [0, 60, 31.5], [0, 0, 1]]
    (tmp_path / "stations.json").write_text(
        json.dumps(
            [
                {
                    "id": "000",
                    "camera_to_world": pose.tolist(),
                    "intrinsics": k,
                    "reference_size": [64, 64],
                }
            ]
        )
    )
    reference = tmp_path / "stations/000/reference.png"
    reference.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), tuple(palette[0])).save(reference)
    output = tmp_path / "horizontal"
    report = export_sweep(
        tmp_path,
        "000",
        output,
        width=64,
        height=64,
        mode="horizontal",
        frame_count=13,
        world_up=[0, -1, 0],
    )
    assert report["coverage_target"] == "horizontal_ring"
    assert report["clean_frame_indices"] == [0, 12]
    assert len(list((output / "frames").glob("*.png"))) == 13
    for i, expected in zip([0, 3, 6, 9], palette):
        actual = np.asarray(Image.open(output / "raw-frames" / f"{i:03d}.png"))
        np.testing.assert_array_equal(actual[32, 32], expected)
    for frame in report["frames"]:
        np.testing.assert_array_equal(
            np.asarray(frame["camera_to_world"])[:3, 3], pose[:3, 3]
        )
    for i in [0, 12]:
        np.testing.assert_array_equal(
            np.asarray(Image.open(output / "frames" / f"{i:03d}.png")),
            np.asarray(Image.open(reference)),
        )
