import numpy as np

from lingbot_map.reconstruction.geometry import preprocessing_transform
from lingbot_map.reconstruction.registration import (
    learned_rotations,
    select_learned_cameras,
)


def test_depth_camera_calibration_preserves_rays_and_shared_sfm_camera(tmp_path):
    sfm_k = np.array([[800.0, 0, 640], [0, 800.0, 360], [0, 0, 1]])
    shared = {"width": 1280, "height": 720, "intrinsics": sfm_k.copy()}
    e = np.column_stack([np.eye(3), np.zeros(3)])
    images = {i: {"camera": shared, "extrinsics": e.copy()} for i in range(2)}
    depth_k = np.array([[430.0, 0, 259], [0, 415.0, 147], [0, 0, 1]])
    path = tmp_path / "000000.npz"
    np.savez(
        path,
        frame_ids=np.array([0, 1]),
        intrinsics=np.stack([depth_k, depth_k * [1.1, 1, 1]]),
        depth=np.ones((2, 294, 518)),
    )
    rotations = {0: np.eye(3), 1: np.eye(3)}
    select_learned_cameras(images, rotations, [path], [(0, 2)])
    pixel = np.array([913.5, 475.5, 1.0])
    affine = preprocessing_transform(1280, 720, 518, 294)
    expected_ray = np.linalg.solve(depth_k, affine @ pixel)
    actual_ray = np.linalg.solve(images[0]["camera"]["intrinsics"], pixel)
    np.testing.assert_allclose(actual_ray, expected_ray)
    np.testing.assert_array_equal(shared["intrinsics"], sfm_k)
    assert images[0]["camera"] is not images[1]["camera"]
    assert not np.array_equal(
        images[0]["camera"]["intrinsics"], images[1]["camera"]["intrinsics"]
    )


def test_owned_window_preserves_relative_orientation_across_overlap(tmp_path):
    from scipy.spatial.transform import Rotation

    first = np.tile(np.column_stack([np.eye(3), np.zeros(3)]), (6, 1, 1))
    second = first.copy()
    second[2:, :, :3] = Rotation.from_rotvec([0, 0.04, 0]).as_matrix()
    a, b = tmp_path / "000000.npz", tmp_path / "000002.npz"
    np.savez(a, frame_ids=np.arange(6), extrinsics=first)
    np.savez(b, frame_ids=np.arange(2, 8), extrinsics=second)
    rotations, _ = learned_rotations(
        [a, b], [(0, 6), (2, 8)], match_depth_ownership=True
    )
    # These two cameras face the same direction in their shared depth window.
    # Keeping frame 4's stale earlier pose would introduce a false relative turn.
    ray = np.array([0.1, 0.2, 1.0])
    np.testing.assert_allclose(rotations[4] @ rotations[7].T @ ray, ray, atol=1e-10)
    np.testing.assert_allclose(rotations[3], np.eye(3), atol=1e-10)


def test_bootstrap_ownership_covers_capture_once_and_rejects_gaps():
    import pytest

    from lingbot_map.reconstruction.geometry import ownership_bounds

    windows = [(0, 96), (72, 168), (144, 193)]
    owned = [
        frame
        for i in range(len(windows))
        for frame in range(*ownership_bounds(windows, i, 8))
    ]
    assert owned == list(range(193))
    with pytest.raises(ValueError, match="overlap"):
        ownership_bounds(windows, 0, 25)


def test_bridge_resume_signature_detects_changed_evidence(tmp_path):
    import json

    from lingbot_map.reconstruction.bridges import bridge_input_signature

    root = tmp_path / "burst"
    (root / "windows").mkdir(parents=True)
    (root / "input.json").write_text("{}")
    (root / "inference.json").write_text("{}")
    archive = root / "windows" / "000000.npz"
    np.savez(archive, depth=np.ones((2, 2)))
    plan = tmp_path / "bridges.json"
    plan.write_text(json.dumps([{"output": "burst", "frames": [1, 2]}]))
    original = bridge_input_signature(plan)
    assert bridge_input_signature(plan) == original
    np.savez(archive, depth=np.full((2, 2), 2.0))
    updated = bridge_input_signature(plan)
    assert original["manifest_sha256"] == updated["manifest_sha256"]
    assert original != updated
