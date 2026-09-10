import json
import subprocess
import sys

import pytest
from PIL import Image

from lingbot_map.reconstruction.images import frame_label, prepare_images


def test_image_cli_preserves_photographs_and_rejects_changed_resume(tmp_path):
    source, output = tmp_path / "photos", tmp_path / "run"
    source.mkdir()
    for name, color in [("frame10.png", (12, 70, 190)), ("frame2.png", (198, 95, 23))]:
        Image.new("RGB", (42, 28), color).save(source / name)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lingbot_map.reconstruction",
            "prepare",
            "--images",
            str(source),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "input.json").read_text())
    assert [row["source_file"] for row in manifest["frames"]] == [
        "frame2.png",
        "frame10.png",
    ]
    for row in manifest["frames"]:
        assert (output / row["file"]).read_bytes() == (
            source / row["source_file"]
        ).read_bytes()
        assert row["timestamp_seconds"] is None
        assert (
            frame_label({"frame": row["id"], "timestamp_seconds": None})
            == f"Frame {row['id']}"
        )
    assert prepare_images(source, output) == manifest
    preserved = (output / "frames/000000.png").read_bytes()
    Image.new("RGB", (42, 28), (0, 0, 0)).save(source / "frame2.png")
    with pytest.raises(ValueError, match="sequence changed"):
        prepare_images(source, output)
    assert (output / "frames/000000.png").read_bytes() == preserved
