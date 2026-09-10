import numpy as np
import pytest

from artifacts.research.experiments.prepare_brush_dataset import nerf_frame


def test_brush_camera_import_preserves_world_projection():
    pose = np.array([[0.0, -1, 0, 3], [1, 0, 0, -2], [0, 0, 1, 1], [0, 0, 0, 1]])
    k = np.array([[200.0, 0, 259], [0, 210.0, 147], [0, 0, 1]])
    frame = nerf_frame(
        {"camera_to_world": pose, "intrinsics": k}, "images/a.png", 518, 294
    )
    imported = np.asarray(frame["transform_matrix"])
    imported[:, 1:3] *= -1  # Brush's actual NeRF importer changes these columns.
    np.testing.assert_array_equal(imported, pose)
    camera_point = np.array([0.2, -0.1, 2, 1])
    world_point = pose @ camera_point
    recovered = np.linalg.inv(imported) @ world_point
    pixel = k @ recovered[:3]
    np.testing.assert_allclose(pixel[:2] / pixel[2], [279, 136.5])
    assert frame["fl_x"] == 200 and frame["fl_y"] == 210


def test_scaled_camera_pose_is_rejected():
    with pytest.raises(ValueError, match="orthonormal"):
        nerf_frame(
            {"camera_to_world": np.diag([2.0, 2, 2, 1]), "intrinsics": np.eye(3)},
            "a.png",
            518,
            294,
        )


def test_small_registration_drift_is_projected_without_moving_camera():
    pose = np.eye(4)
    pose[0, 1] = 0.00018  # Observed registration drift exceeds the old 1e-4 gate.
    pose[:3, 3] = [3, -2, 1]
    original = pose.copy()
    frame = nerf_frame(
        {"camera_to_world": pose, "intrinsics": np.eye(3)}, "a.png", 518, 294
    )
    restored = np.asarray(frame["transform_matrix"]) @ np.diag([1, -1, -1, 1])
    np.testing.assert_allclose(
        restored[:3, :3].T @ restored[:3, :3], np.eye(3), atol=1e-12
    )
    np.testing.assert_array_equal(restored[:3, 3], original[:3, 3])
    np.testing.assert_array_equal(pose, original)
    assert np.max(np.abs(restored - original)) < 1e-4
