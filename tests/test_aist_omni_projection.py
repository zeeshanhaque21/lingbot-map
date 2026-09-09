import numpy as np

from artifacts.research.experiments.compare_panorama_sources import sample_panorama
from artifacts.research.experiments.prepare_aist_omni_pilot import aist_mask, projection


def test_panorama_cardinals_and_vertical_direction():
    # Independent landmarks on a 360 x 180 panorama: back, left, forward, right.
    panorama = np.zeros((180, 360, 3), np.float32)
    for column, color in [(0, [1, 1, 0]), (90, [0, 1, 0]),
                          (180, [0, 0, 1]), (270, [1, 0, 0])]:
        panorama[:, (column + np.arange(-4, 4)) % 360] = color
    for yaw, expected in [(0, [0, 0, 1]), (90, [1, 0, 0]),
                          (180, [1, 1, 0]), (270, [0, 1, 0])]:
        _, _, x, y = projection(9, 9, 60, yaw, panorama.shape)
        view = sample_panorama(panorama, x, y)
        np.testing.assert_allclose(view[4, 4], expected)
        assert y[0, 4] < 89.5 < y[-1, 4]


def test_pilot_views_exclude_supplied_aist_invalid_regions():
    mask = aist_mask((960, 1920))
    assert mask[959, 960] == 0
    assert mask[10, 960] == 0
    assert mask[720, 100] == 0
    assert mask[480, 960] == 1
    for yaw in [0, 90, 180, 270]:
        intrinsic, rotation, x, y = projection(518, 294, 100, yaw, mask.shape)
        assert np.all(sample_panorama(mask, x, y) > 0.999)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(intrinsic[:2, 2], [258.5, 146.5])
