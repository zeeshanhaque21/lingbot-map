import json

from PIL import Image

from artifacts.research.experiments.run_fixanything_mac import inspect_sweep


def make_sweep(tmp_path):
    (tmp_path / "frames").mkdir()
    frames = []
    for i in range(61):
        frames.append({"captured_anchor": i in (0, 30, 60), "observed_fraction": 0.9})
        color = (60, 80, 100) if i in (0, 60) else (110, 130, 150)
        Image.new("RGB", (64, 64), color).save(tmp_path / "frames" / f"{i:03d}.png")
    config = {
        "frames": frames,
        "clean_frame_indices": [0, 30, 60],
        "width": 64,
        "height": 64,
    }
    (tmp_path / "sweep.json").write_text(json.dumps(config))
    return config


def test_unobserved_direction_and_partial_clip_cannot_pass_pilot_check(tmp_path):
    config = make_sweep(tmp_path)
    config["frames"][31]["observed_fraction"] = 0
    (tmp_path / "sweep.json").write_text(json.dumps(config))
    result = inspect_sweep(tmp_path, 61)
    assert not result["ready_for_pilot"]
    assert result["empty_render_indices"] == [31]
    assert not inspect_sweep(tmp_path, 5)["ready_for_pilot"]


def test_middle_anchor_is_preserved_and_repeated_endpoint_is_not_new_evidence(tmp_path):
    make_sweep(tmp_path)
    result = inspect_sweep(tmp_path, 61)
    assert result["ready_for_pilot"]
    assert result["clean_frame_indices"] == [0, 30, 60]
    assert result["unique_clean_images"] == 2
    assert not result["panorama_quality_verified"]
