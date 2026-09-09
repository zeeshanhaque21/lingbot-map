import numpy as np

from artifacts.research.experiments.gaussian_depth_renderer import (
    project_gaussians,
    rasterize,
)


def scene(means, scales, opacities, colors=None, quaternions=None):
    count = len(means)
    colors = np.ones((count, 3)) if colors is None else np.asarray(colors)
    return {
        "means": np.asarray(means, np.float32),
        "log_scales": np.log(np.asarray(scales, np.float32)),
        "raw_opacity": np.log(
            np.asarray(opacities, np.float32) / (1 - np.asarray(opacities, np.float32))
        ),
        "quaternions": np.asarray(quaternions, np.float32)
        if quaternions is not None
        else np.tile([1, 0, 0, 0], (count, 1)).astype(np.float32),
        "coefficients": ((colors.astype(np.float32) - 0.5) / 0.28209479177387814)[
            :, None
        ],
    }


def draw(gaussians):
    k = np.array([[20.0, 0, 8.5], [0, 20.0, 8.5], [0, 0, 1]], np.float32)
    projected, extents = project_gaussians(gaussians, np.eye(4), k, 17, 17)
    return rasterize(projected, extents, 17, 17), projected


def test_front_to_back_compositing_and_depth_moments():
    result, _ = draw(
        scene(
            [[0, 0, 4], [0, 0, 2]],
            [[0.2, 0.2, 0.01]] * 2,
            [0.5, 0.5],
            [[0, 0, 1], [1, 0, 0]],
        )
    )
    np.testing.assert_allclose(result["rgb"][8, 8], [0.5, 0, 0.25], atol=1e-6)
    np.testing.assert_allclose(result["alpha"][8, 8], 0.75, atol=1e-6)
    np.testing.assert_allclose(result["center_depth"][8, 8], 8 / 3, atol=1e-6)
    assert result["center_sigma"][8, 8] > 0.9


def test_conditional_depth_respects_tilted_thin_gaussian():
    angle = np.pi / 8
    q = [[np.cos(angle), 0, np.sin(angle), 0]]
    result, projected = draw(
        scene([[0, 0, 2]], [[0.3, 0.3, 0.001]], [0.95], quaternions=q)
    )
    assert abs(result["depth"][8, 8] - 2) < 1e-6
    assert result["depth"][8, 10] < 1.83
    assert result["depth"][8, 6] > 2.17
    assert projected[0, 12] < projected[0, 13] / 10


def test_behind_camera_and_transparent_gaussians_contribute_nothing():
    result, projected = draw(
        scene([[0, 0, -2], [0, 0, 2]], [[0.2, 0.2, 0.1]] * 2, [0.99, 0.001])
    )
    assert len(projected) == 0
    assert np.count_nonzero(result["rgb"]) == 0
    assert np.count_nonzero(result["alpha"]) == 0


def test_brush_termination_excludes_gaussian_crossing_transmittance_floor():
    result, _ = draw(
        scene(
            [[0, 0, 2], [0, 0, 3]],
            [[0.2, 0.2, 0.1]] * 2,
            [0.999, 0.999],
            [[1, 0, 0], [0, 0, 1]],
        )
    )
    np.testing.assert_allclose(result["rgb"][8, 8], [0.999, 0, 0], atol=1e-6)
    np.testing.assert_allclose(result["center_depth"][8, 8], 2, atol=1e-6)
