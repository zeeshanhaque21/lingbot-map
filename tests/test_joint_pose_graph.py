import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from artifacts.research.experiments.joint_pose_graph import information_matrix, optimize


def test_information_matches_open3d_point_correspondences():
    points = np.random.default_rng(80).normal(size=(60, 3))
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    expected = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        cloud, cloud, 0.01, np.eye(4)
    )
    np.testing.assert_allclose(information_matrix(points, 1.0), expected, atol=1e-10)


def test_joint_optimizer_corrects_known_camera_errors(tmp_path):
    world = np.random.default_rng(36).uniform(
        [-0.6, -0.4, 2.0], [0.6, 0.4, 3.0], (60, 3)
    )
    true = np.repeat(np.eye(4)[None], 3, axis=0)
    for i in range(3):
        true[i, :3, :3] = Rotation.from_rotvec([0.0, i * 0.04, 0.0]).as_matrix()
        true[i, :3, 3] = [i * 0.2, 0.0, 0.0]
    initial = true.copy()
    initial[1:, :3, 3] += [0.08, 0.06, 0.04]
    initial[1:, :3, :3] = (
        Rotation.from_rotvec([0.02, -0.03, 0.01]).as_matrix() @ initial[1:, :3, :3]
    )
    k = np.array([[300.0, 0.0, 259.0], [0.0, 300.0, 147.0], [0.0, 0.0, 1.0]])
    cameras, frames, tracks, xyz, pixels = [], [], [], [], []
    for i in range(3):
        points = (world - true[i, :3, 3]) @ true[i, :3, :3]
        projected = points @ k.T
        cameras.append(
            {
                "frame": i,
                "camera_to_world": initial[i].tolist(),
                "intrinsics": k.tolist(),
            }
        )
        frames.extend([i] * len(world))
        tracks.extend(range(len(world)))
        xyz.extend(points)
        pixels.extend(projected[:, :2] / projected[:, 2:])
    data = {
        "frame": np.array(frames),
        "track": np.array(tracks),
        "xyz": np.array(xyz),
        "pixel": np.array(pixels),
    }
    edges = []
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        target = (world - true[b, :3, 3]) @ true[b, :3, :3]
        edges.append(
            {
                "frames": [a, b],
                "accepted": True,
                "transform": (np.linalg.inv(true[b]) @ true[a]).tolist(),
                "information": information_matrix(target, 0.01).tolist(),
            }
        )
    result = optimize(data, cameras, edges, tmp_path, prior_weight=0.001)
    np.testing.assert_allclose(result[0], true[0], atol=1e-8)
    assert np.max(np.abs(result - true)) < 0.002
    np.testing.assert_allclose(np.linalg.det(result[:, :3, :3]), 1.0, atol=1e-8)
