import numpy as np

from lingbot_map.reconstruction.source_photos import project_photos


def test_translated_source_photo_uses_parallax_and_rejects_occlusion():
    k = np.array([[4.0, 0, 4], [0, 4, 4], [0, 0, 1]])
    ext = np.eye(4)
    ext[0, 3] = -1  # A source camera translated one world unit to the right.
    rgb = np.zeros((9, 9, 3), np.uint8)
    rgb[:, :, 0] = np.arange(9)[None, :] * 20
    photo = {
        "frame": 17,
        "rgb": rgb,
        "depth": np.full((9, 9), 2, np.float32),
        "intrinsics": k,
        "photo_intrinsics": k,
        "world_to_camera": ext,
    }
    world = np.array([[[0.0, 0, 2], [0.0, 0, 4], [0.0, 0, 2]]])
    base = np.full((1, 3, 3), 99, np.uint8)
    output, provenance = project_photos(
        world, np.array([[True, True, False]]), base, np.zeros(3), [photo]
    )
    # The visible point projects to source x=2, rather than the target's x=4.
    np.testing.assert_array_equal(output[0, 0], [40, 0, 0])
    np.testing.assert_array_equal(output[0, 1:], base[0, 1:])
    np.testing.assert_array_equal(provenance, [[17, -1, -1]])


def test_photo_projection_batches_large_frame_and_keeps_empty_baseline():
    k = np.array([[4.0, 0, 4], [0, 4, 4], [0, 0, 1]])
    world = np.broadcast_to([0.0, 0, 2], (256, 256, 3))
    base = np.full((256, 256, 3), 99, np.uint8)
    visible = np.ones((256, 256), bool)
    unchanged, provenance = project_photos(world, visible, base, np.zeros(3), [])
    np.testing.assert_array_equal(unchanged, base)
    assert (provenance == -1).all()
    photo = {
        "frame": 1,
        "rgb": np.full((9, 9, 3), 41, np.uint8),
        "depth": np.full((9, 9), 2, np.float32),
        "intrinsics": k,
        "photo_intrinsics": k,
        "world_to_camera": np.eye(4),
    }
    output, provenance = project_photos(world, visible, base, np.zeros(3), [photo])
    assert (output == 41).all()
    assert (provenance == 1).all()
