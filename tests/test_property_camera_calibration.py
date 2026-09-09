import numpy as np

from lingbot_map.reconstruction.geometry import preprocessing_transform
from lingbot_map.reconstruction.registration import select_learned_cameras


def test_depth_camera_calibration_preserves_rays_and_shared_sfm_camera(tmp_path):
    sfm_k = np.array([[800.0, 0, 640], [0, 800.0, 360], [0, 0, 1]])
    shared = {"width": 1280, "height": 720, "intrinsics": sfm_k.copy()}
    e = np.column_stack([np.eye(3), np.zeros(3)])
    images = {i: {"camera": shared, "extrinsics": e.copy()} for i in range(2)}
    depth_k = np.array([[430.0, 0, 259], [0, 415.0, 147], [0, 0, 1]])
    path = tmp_path / "000000.npz"
    np.savez(path, frame_ids=np.array([0, 1]), intrinsics=np.stack([depth_k, depth_k * [1.1, 1, 1]]), depth=np.ones((2, 294, 518)))
    rotations = {0: np.eye(3), 1: np.eye(3)}
    select_learned_cameras(images, rotations, [path], [(0, 2)])
    pixel = np.array([913.5, 475.5, 1.0])
    affine = preprocessing_transform(1280, 720, 518, 294)
    expected_ray = np.linalg.solve(depth_k, affine @ pixel)
    actual_ray = np.linalg.solve(images[0]["camera"]["intrinsics"], pixel)
    np.testing.assert_allclose(actual_ray, expected_ray)
    np.testing.assert_array_equal(shared["intrinsics"], sfm_k)
    assert images[0]["camera"] is not images[1]["camera"]
    assert not np.array_equal(images[0]["camera"]["intrinsics"], images[1]["camera"]["intrinsics"])
