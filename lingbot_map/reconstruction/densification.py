"""Recover temporal evidence around photogrammetry breaks."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from .colmap_io import read_model
from .io import digest, write_json


def densify(source, destination):
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source / "input.json").read_text())
    if manifest["configuration"]["fps"] != 2:
        raise ValueError("Adaptive densification currently expects the 2 fps baseline")
    if destination.exists():
        raise ValueError("Densification output exists; choose a new directory")
    owners = {i: set() for i in range(len(manifest["frames"]))}
    for model in (source / "colmap/sparse").glob("*/text"):
        images, points = read_model(model)
        if len(points) < 100:
            continue
        for name in images:
            owners[int(Path(name).stem)].add(model.parent.name)
    if not any(owners.values()):
        raise ValueError("Run sfm before selecting capture breaks")
    cuts = [i for i in range(len(owners) - 1) if not owners[i] & owners[i + 1]]
    groups = []
    for index in cuts:
        if not groups or index - groups[-1][-1] > 6:
            groups.append([index])
        else:
            groups[-1].append(index)
    intervals = [(max(0, g[0] / 2 - 2), (g[-1] + 1) / 2 + 2) for g in groups]
    if not intervals:
        raise ValueError("No temporal registration breaks found")
    destination.mkdir(parents=True)
    frames = destination / "frames"
    samples = destination / "supplemental"
    frames.mkdir()
    samples.mkdir()
    for frame in manifest["frames"]:
        original = source / frame["file"]
        (frames / original.name).symlink_to(original.resolve())
    selection = "+".join(f"between(t,{lo},{hi})" for lo, hi in intervals)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n"]
    if sys.platform == "darwin":
        command += ["-hwaccel", "videotoolbox"]
    command += [
        "-i",
        manifest["configuration"]["source"],
        "-vf",
        f"fps=10:start_time=0,select='{selection}',scale=1280:-2",
        "-fps_mode",
        "vfr",
        "-frame_pts",
        "1",
        "-q:v",
        "2",
        str(samples / "%06d.jpg"),
    ]
    subprocess.run(command, check=True)
    supplemental = []
    for sample in sorted(samples.glob("*.jpg")):
        tick = int(sample.stem)
        # At 50 fps input, the default fps filter's 2 fps frame i selects
        # the same source frame as 10 fps tick 5*i+2. Retain temporal ordering.
        base, offset = divmod(tick - 2, 5)
        if offset == 0 or base < 0 or base >= len(owners) - 1:
            continue
        name = f"{base:06d}_{offset}.jpg"
        (frames / name).symlink_to(sample.resolve())
        supplemental.append(
            {"file": f"frames/{name}", "output_timestamp_seconds": tick / 10}
        )
    manifest["supplemental"] = supplemental
    manifest["densification"] = {
        "intervals_seconds": intervals,
        "fps": 10,
        "source_manifest_sha256": digest(source / "input.json"),
    }
    write_json(destination / "input.json", manifest)
    database = destination / "colmap/database.db"
    database.parent.mkdir()
    with (
        sqlite3.connect(source / "colmap/database.db") as old,
        sqlite3.connect(database) as new,
    ):
        old.backup(new)
    write_json(
        destination / "densification.json",
        {
            "extra_frames": len(supplemental),
            "baseline_frames": len(owners),
            "intervals_seconds": intervals,
            "source": str(source.resolve()),
        },
    )
