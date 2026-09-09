import numpy as np
import torch
from scipy.spatial.transform import Rotation

from artifacts.research.experiments.direct_bundle_refinement import (
    BundleProblem,
    fit_bundle,
)


def known_scene():
    world = np.random.default_rng(17).uniform(
        [-0.6, -0.4, 2.0], [0.6, 0.4, 3.0], (60, 3)
    )
    true = np.repeat(np.eye(4)[None], 3, axis=0)
    k = np.array([[300.0, 0, 259], [0, 300.0, 147], [0, 0, 1.0]])
    cameras, frames, tracks, xyz, pixels = [], [], [], [], []
    for i in range(3):
        true[i, :3, :3] = Rotation.from_rotvec([0, i * 0.04, 0]).as_matrix()
        true[i, :3, 3] = [i * 0.2, 0, 0]
        initial = true[i].copy()
        if i:
            initial[:3, 3] += [0.08, 0.04, 0.02]
            initial[:3, :3] = (
                initial[:3, :3] @ Rotation.from_rotvec([0.01, -0.02, 0.01]).as_matrix()
            )
        points = (world - true[i, :3, 3]) @ true[i, :3, :3]
        projected = points @ k.T
        cameras.append(
            {"frame": i, "camera_to_world": initial.tolist(), "intrinsics": k.tolist()}
        )
        frames.extend([i] * len(world))
        tracks.extend(range(len(world)))
        xyz.extend(points)
        pixels.extend(projected[:, :2] / projected[:, 2:])
    return (
        {
            "frame": np.array(frames),
            "track": np.array(tracks),
            "xyz": np.array(xyz),
            "pixel": np.array(pixels),
        },
        cameras,
        true,
    )


def test_reserved_tracks_do_not_change_bundle_objective():
    data, cameras, _ = known_scene()
    reference = BundleProblem(data, cameras)
    changed = {key: value.copy() for key, value in data.items()}
    held = changed["track"] % 5 == 0
    changed["xyz"][held] *= 1000
    changed["pixel"][held] += 10000
    candidate = BundleProblem(changed, cameras)
    assert len(candidate.landmarks) == 48
    assert reference.unit == candidate.unit
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(
            candidate.state_dict()[name], expected, rtol=0, atol=0
        )


def test_bundle_recovers_known_motion_and_finite_zero_rotation_gradient(tmp_path):
    data, cameras, true = known_scene()
    problem = BundleProblem(data, cameras)
    initial_loss = float(problem.loss_terms()[0].detach())
    problem.loss_terms()[0].backward()
    assert all(torch.isfinite(p.grad).all() for p in problem.parameters())
    poses, scales = fit_bundle(problem, tmp_path, steps=8)
    assert float(problem.loss_terms()[0].detach()) < initial_loss * 0.1
    np.testing.assert_allclose(poses[0], true[0], atol=1e-6)
    assert np.max(np.abs(poses - true)) < 0.025
    np.testing.assert_allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=2e-6)
    assert np.max(np.abs(scales - 1)) < 0.025


def test_cauchy_bundle_retains_geometry_with_gross_feature_outliers(tmp_path):
    data, cameras, true = known_scene()
    corrupt = (data["frame"] == 1) & np.isin(data["track"], [1, 2, 3])
    data["pixel"][corrupt] += 5000
    problem = BundleProblem(data, cameras, pixel_loss="cauchy")
    poses, scales = fit_bundle(problem, tmp_path, steps=8)
    assert np.max(np.abs(poses - true)) < 0.025
    assert np.max(np.abs(scales - 1)) < 0.025
