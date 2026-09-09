"""Preserve and normalize archives from the first local experiment."""

import json
from pathlib import Path

import numpy as np

from .io import digest, write_json, write_npz
from .poses import LONG_CHECKPOINT_SHA256, world_to_camera


def normalize_archives(source, destination):
    source, destination = Path(source), Path(destination)
    configuration = json.loads((source / "inference.json").read_text())
    if (
        configuration.get("schema") != 1
        or configuration["checkpoint_sha256"] != LONG_CHECKPOINT_SHA256
    ):
        raise ValueError(
            "Normalization only accepts the original experiment with the verified long checkpoint"
        )
    if destination.exists():
        raise ValueError("Normalization output exists; choose a new directory")
    manifest = json.loads((source / "input.json").read_text())
    files = sorted((source / "windows").glob("*.npz"))
    from .inference import window_ranges

    expected = list(
        window_ranges(
            len(manifest["frames"]), configuration["window"], configuration["overlap"]
        )
    )
    if [int(f.stem) for f in files] != [start for start, _ in expected]:
        raise ValueError("Finish source inference before normalizing its archives")
    destination.mkdir(parents=True)
    (destination / "frames").symlink_to(
        (source / "frames").resolve(), target_is_directory=True
    )
    write_json(destination / "input.json", manifest)
    configuration.update(
        {
            "schema": 2,
            "checkpoint_pose_convention": "camera-to-world",
            "stored_pose_convention": "world-to-camera",
            "derived_from": str(source.resolve()),
        }
    )
    write_json(destination / "inference.json", configuration)
    for file in files:
        data = dict(np.load(file))
        data["extrinsics"] = world_to_camera(data["extrinsics"], "camera-to-world")
        destination_file = destination / "windows" / file.name
        write_npz(destination_file, **data)
        write_json(
            destination_file.with_suffix(".json"),
            {
                "source_sha256": digest(file),
                "sha256": digest(destination_file),
                "operation": "camera-to-world to world-to-camera",
            },
        )
