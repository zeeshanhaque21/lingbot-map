"""Adapt video sampling at disconnected camera joins, preserving every attempt."""

import json
import platform
import subprocess
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .io import write_json


def disconnected_joins(diagnostics, frame_count):
    edges = np.asarray(diagnostics["edge_frames"], int)
    if not len(edges):
        raise ValueError(
            "No supported pose edges; capture overlapping views before repair"
        )
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(frame_count, frame_count),
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    frames = np.flatnonzero(sizes[labels] >= 3)
    candidates = [
        (int(a), int(b))
        for a, b in pairwise(frames)
        if labels[a] != labels[b] and b - a <= 3
    ]
    parent = list(range(len(sizes)))

    def root(label):
        while parent[label] != label:
            label = parent[label]
        return label

    chosen = []
    for a, b in sorted(candidates, key=lambda pair: (pair[1] - pair[0], pair[0])):
        left, right = root(int(labels[a])), root(int(labels[b]))
        if left != right:
            parent[right] = left
            chosen.append((a, b))
    if not chosen:
        raise ValueError(
            "Disconnected observations have no short video bridge; additional capture is required"
        )
    return sorted(chosen)


def prepare_motion_bridge(source, output, a, b):
    source, output = Path(source), Path(output)
    manifest = json.loads((source / "input.json").read_text())
    if manifest["configuration"]["fps"] != 2:
        raise ValueError(
            "Adaptive 10 fps bridges currently require a 2 fps base reconstruction"
        )
    start, end = max(0, a - 4), min(len(manifest["frames"]) - 1, b + 4)
    config = {
        **manifest["configuration"],
        "fps": 10,
        "selection": [start, end],
        "bridge": [a, b],
        "sampling_schema": 2,
    }
    marker = output / "input.json"
    if marker.exists():
        existing = json.loads(marker.read_text())
        if existing["configuration"] != config or not all(
            (output / f["file"]).exists() for f in existing["frames"]
        ):
            raise ValueError(
                "Motion bridge cache differs from the requested capture interval"
            )
        return
    output.mkdir(parents=True, exist_ok=True)
    decoded = output / "decoded"
    decoded.mkdir(exist_ok=True)
    count = (end - start) * 5 + 1
    existing = sorted(decoded.glob("*.jpg"))
    if [int(p.stem) for p in existing] != list(range(len(existing))):
        raise ValueError(
            "Partial bridge extraction has gaps; preserve it and use a new output"
        )
    if len(existing) < count:
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n"]
        if platform.system() == "Darwin":
            command += ["-hwaccel", "videotoolbox"]
        command += [
            "-ss",
            str(start / 2),
            "-i",
            manifest["configuration"]["source"],
            "-vf",
            f"fps=10:start_time=0,trim=start_frame={2 + len(existing)},scale=1280:-2",
            "-fps_mode",
            "vfr",
            "-frames:v",
            str(count - len(existing)),
            "-q:v",
            "2",
            "-start_number",
            str(len(existing)),
            str(decoded / "%06d.jpg"),
        ]
        with (output / "extraction.log").open("a") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if len(list(decoded.glob("*.jpg"))) != count:
        raise ValueError("Video ended before the requested motion bridge")
    (output / "frames").mkdir(exist_ok=True)
    frames = []
    for index in range(count):
        original = start + index // 5 if index % 5 == 0 else None
        path = output / "frames" / f"{index:06d}.jpg"
        origin = (
            source / manifest["frames"][original]["file"]
            if original is not None
            else decoded / f"{index:06d}.jpg"
        )
        if not path.exists():
            path.symlink_to(origin.resolve())
        frames.append(
            {
                "id": index,
                "file": str(path.relative_to(output)),
                "timestamp_seconds": start / 2 + 0.2 + index / 10,
                "original_frame": original,
            }
        )
    write_json(
        marker,
        {
            "configuration": config,
            "probe": manifest["probe"],
            "coordinate_convention": manifest["coordinate_convention"],
            "units": manifest["units"],
            "frames": frames,
        },
    )


def register_with_repairs(
    source, destination, model, checkpoint, bridges=None, maximum_rounds=2,
    camera_source="sfm",
):
    from .inference import infer
    from .registration import register

    source, destination = Path(source), Path(destination)
    configuration = json.loads((source / "inference.json").read_text())
    manifest = json.loads((source / "input.json").read_text())
    specifications = []
    if bridges:
        bridges = Path(bridges)
        specifications = [
            {**item, "output": str((bridges.parent / item["output"]).resolve())}
            for item in json.loads(bridges.read_text())
        ]
    for attempt in range(maximum_rounds + 1):
        target = (
            destination
            if attempt == 0
            else destination.with_name(f"{destination.name}-repair-{attempt}")
        )
        try:
            register(source, target, model, bridges, camera_source=camera_source)
            write_json(
                source / "pipeline-result.json",
                {
                    "output": str(target.resolve()),
                    "motion_bridges": str(Path(bridges).resolve()) if bridges else None,
                    "registration_attempts": attempt + 1,
                },
            )
            return target
        except ValueError as error:
            diagnostic = target / "registration-diagnostics.json"
            if (
                "connects only" not in str(error)
                or not diagnostic.exists()
                or attempt == maximum_rounds
            ):
                raise
            joins = disconnected_joins(
                json.loads(diagnostic.read_text()), len(manifest["frames"])
            )
            for a, b in joins:
                if any(item["frames"] == [a, b] for item in specifications):
                    raise ValueError(
                        f"Dense bridge {a}-{b} remains disconnected; inspect capture and motion diagnostics"
                    ) from error
                output = source / "motion-bridges-v2" / f"{a:06d}-{b:06d}"
                prepare_motion_bridge(source, output, a, b)
                infer(
                    output,
                    checkpoint,
                    device=configuration["device"],
                    precision=configuration["precision"],
                )
                specifications.append(
                    {"output": str(output.resolve()), "frames": [a, b]}
                )
            bridges = source / "motion-bridges-v2" / f"bridges-{attempt + 1}.json"
            if bridges.exists() and json.loads(bridges.read_text()) != specifications:
                raise ValueError(
                    "Repair plan changed; preserve the existing run and use a new output"
                )
            write_json(bridges, specifications)
    raise RuntimeError("Registration repair did not produce a model")
