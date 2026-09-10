import numpy as np
import open3d as o3d

from artifacts.research.experiments.compare_panorama_sources import (
    color_error,
    sample_panorama,
    source_coordinates,
)
from lingbot_map.reconstruction.rendering import SurfaceRenderer


def test_reprojection_rejects_surface_hidden_from_station():
    wall = o3d.geometry.TriangleMesh.create_box(8, 8, 0.1)
    wall.translate((-4, -4, 3))
    wall.paint_uniform_color((0.5, 0.5, 0.5))
    camera = {
        "intrinsics": [[40, 0, 32], [0, 40, 24], [0, 0, 1]],
        "camera_to_world": np.eye(4),
    }
    station = np.eye(4)
    station[0, 3] = 1
    x, y, mask = source_coordinates(
        SurfaceRenderer(wall), camera, station, 65, 49, (360, 180)
    )
    assert mask[24, 32]
    # Source center sees (0, 0, 3); station center is displaced one unit right.
    assert np.isclose(x[24, 32], 179.5 - np.degrees(np.arctan(1 / 3)), atol=1e-4)
    assert np.isclose(y[24, 32], 89.5)
    occluder = o3d.geometry.TriangleMesh.create_box(0.25, 4, 0.3)
    occluder.translate((0.55, -2, 0.9))
    occluder.paint_uniform_color((1, 0, 0))
    _, _, obstructed = source_coordinates(
        SurfaceRenderer(wall + occluder), camera, station, 65, 49, (360, 180)
    )
    assert not obstructed[24, 32]


def test_spherical_sampling_wraps_longitude_without_wrapping_poles():
    panorama = np.zeros((4, 8, 3), np.uint8)
    panorama[0] = [0, 200, 0]
    panorama[-1] = [0, 0, 200]
    panorama[1, 0] = [100, 0, 0]
    panorama[1, -1] = [200, 0, 0]
    x = np.array([[-0.5, 7.5, 2, 2]], np.float32)
    y = np.array([[1, 1, -0.5, 3.5]], np.float32)
    samples = sample_panorama(panorama, x, y)
    assert np.array_equal(
        samples[0], [[150, 0, 0], [150, 0, 0], [0, 200, 0], [0, 0, 200]]
    )
    mask = np.ones((1, 4), bool)
    assert color_error(samples, samples, mask)["absolute_rgb_error"] == 0
    assert color_error(255 - samples, samples, mask)["absolute_rgb_error"] > 0.5
    assert color_error(samples, samples, ~mask) is None
