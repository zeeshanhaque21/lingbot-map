import numpy as np

from artifacts.research.experiments.photometric_depth_falsifier import refine_frame


def plane_scene(pure_rotation=False, texture=True):
    """Render a physical Z=2 plane analytically, independently of the depth warp."""
    height, width = 48, 64
    focal = 70
    y, x = np.indices((height, width), dtype=float)
    centers = np.zeros(5) if pure_rotation else np.array([-0.24, -0.13, 0, 0.11, 0.22])
    angles = np.array([-0.09, -0.045, 0, 0.04, 0.08]) if pure_rotation else np.zeros(5)
    images = []
    extrinsics = []
    for center, angle in zip(centers, angles):
        cosine, sine = np.cos(angle), np.sin(angle)
        ray_x = cosine * (x - width / 2) / focal + sine
        ray_y = (y - height / 2) / focal
        ray_z = -sine * (x - width / 2) / focal + cosine
        u = focal * (center / 2 + ray_x / ray_z) + width / 2
        v = focal * ray_y / ray_z + height / 2
        rotation = np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
        extrinsics.append(
            np.column_stack([rotation.T, -rotation.T @ np.array([center, 0, 0])])
        )
        if texture:
            rgb = np.stack(
                [
                    0.5
                    + 0.25 * np.sin(0.21 * u + 0.13 * v)
                    + 0.18 * np.sin(0.73 * u - 0.27 * v),
                    0.5
                    + 0.21 * np.sin(0.33 * u - 0.18 * v)
                    + 0.16 * np.cos(0.11 * u + 0.39 * v),
                    0.5
                    + 0.23 * np.cos(0.45 * u + 0.25 * v)
                    + 0.19 * np.sin(0.17 * u - 0.55 * v),
                ],
                -1,
            )
        else:
            rgb = np.full((height, width, 3), 0.5)
        images.append(np.round(rgb.clip(0, 1) * 255).astype(np.uint8))
    depth = np.full((5, height, width), 2, dtype=np.float32)
    depth[2] = 3  # Incorrect prior in the reference view.
    return {
        "frame_ids": np.arange(5),
        "rgb": np.stack(images),
        "depth": depth,
        "confidence": np.ones_like(depth),
        "intrinsics": np.repeat(
            np.array(
                [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
                dtype=np.float32,
            )[None],
            5,
            axis=0,
        ),
        "extrinsics": np.array(extrinsics, dtype=np.float32),
    }


def test_recovers_known_depth_even_when_prior_is_occluded():
    data = plane_scene()
    original = data["depth"].copy()
    result, report = refine_frame(data, 2, "cpu", hypotheses=65)
    accepted = result != original[2]
    assert report["changed_fraction"] > 0.05
    assert report["accepted_from_occluded_prior_fraction"] > 0.05
    assert np.median(np.abs(result[accepted] - 2)) < 0.06
    np.testing.assert_array_equal(data["depth"], original)


def test_unobservable_depth_and_flat_texture_preserve_prior():
    for data in [plane_scene(pure_rotation=True), plane_scene(texture=False)]:
        result, report = refine_frame(data, 2, "cpu", hypotheses=17)
        np.testing.assert_array_equal(result, data["depth"][2])
        assert report["changed_fraction"] == 0


def test_reserved_source_images_cannot_change_training_correction():
    data = plane_scene()
    data["frame_ids"][4] = 5
    first, _ = refine_frame(data, 2, "cpu", hypotheses=33)
    data["rgb"][4] = 255 - data["rgb"][4]
    second, report = refine_frame(data, 2, "cpu", hypotheses=33)
    np.testing.assert_array_equal(first, second)
    assert 5 not in report["source_frames"]


def test_reserved_reference_depth_is_unchanged():
    data = plane_scene()
    data["frame_ids"][2] = 5
    result, report = refine_frame(data, 2, "cpu", hypotheses=33)
    np.testing.assert_array_equal(result, data["depth"][2])
    assert report["changed_fraction"] == 0
