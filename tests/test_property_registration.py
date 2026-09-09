import numpy as np
import pytest

from lingbot_map.reconstruction.registration import solve_centers, track_edges


def test_depth_tracks_recover_translation_with_outlier_observations():
    rng = np.random.default_rng(17)
    truth = np.column_stack(
        [np.arange(8) * 0.1, np.sin(np.arange(8)) * 0.05, np.zeros(8)]
    )
    tracks = {}
    for track in range(150):
        point = rng.uniform([-1, -1, 2], [1, 1, 4])
        observations = []
        for frame in range(8):
            offset = point - truth[frame] + rng.normal(0, 0.002, 3)
            if track % 17 == 1:
                offset += rng.normal(0, 1, 3)
            observations.append((frame, offset, np.zeros(2)))
        tracks[track] = observations
    edges = track_edges(tracks, dict.fromkeys(range(8), 3.0))
    centers, connected, _ = solve_centers(list(range(8)), edges)
    assert connected.all()
    np.testing.assert_allclose(centers, truth, atol=0.004)


def test_disconnected_rooms_fail_instead_of_being_stacked_at_origin():
    edges = [
        (0, 1, np.array([1.0, 0, 0]), 10.0, 1.0),
        (2, 3, np.array([1.0, 0, 0]), 10.0, 1.0),
    ]
    with pytest.raises(ValueError, match="connects only"):
        solve_centers([0, 1, 2, 3], edges)
