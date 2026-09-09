"""Use densely sampled, depth-checked motion across difficult video turns."""

import json
from pathlib import Path

import numpy as np


def load_bridges(path, source, files, ranges, learned):
    from .registration import motion_constraint

    path, source = Path(path), Path(source)
    result = []
    for specification in sorted(
        json.loads(path.read_text()), key=lambda item: item["frames"][0]
    ):
        root = (path.parent / specification["output"]).resolve()
        a, b = specification["frames"]
        manifest = json.loads((root / "input.json").read_text())
        source_manifest = json.loads((source / "input.json").read_text())
        if (
            manifest["configuration"]["source_sha256"]
            != source_manifest["configuration"]["source_sha256"]
        ):
            raise ValueError("Motion bridge belongs to a different source video")
        windows = sorted((root / "windows").glob("*.npz"))
        if len(windows) != 1:
            raise ValueError("A motion bridge must fit in one inference window")
        data = dict(np.load(windows[0]))
        original = {
            f["original_frame"]: f["id"]
            for f in manifest["frames"]
            if f.get("original_frame") is not None
        }
        ia, ib = original[a], original[b]
        if not ia < ib:
            raise ValueError("Motion bridge endpoints must be in capture order")
        h, w = data["depth"][0].shape
        images = {
            i: {
                "extrinsics": data["extrinsics"][i],
                "camera": {
                    "width": w,
                    "height": h,
                    "intrinsics": data["intrinsics"][i],
                },
            }
            for i in range(len(data["frame_ids"]))
        }
        checks = [
            motion_constraint(data, i, i + 1, images, 1.0)[1] for i in range(ia, ib)
        ]
        if not all(c["accepted"] for c in checks):
            raise ValueError(
                f"Dense motion bridge {a}-{b} lacks a continuous depth-supported path"
            )
        owner = next(
            i
            for i, (start, end) in enumerate(ranges)
            if (start if i == 0 else (start + ranges[i - 1][1]) // 2)
            <= a
            < (end if i == len(ranges) - 1 else (end + ranges[i + 1][0]) // 2)
        )
        with np.load(files[owner]) as raw:
            index = int(np.flatnonzero(raw["frame_ids"] == a)[0])
            reference = raw["depth"][index]
        if reference.shape != data["depth"][ia].shape:
            raise ValueError("Bridge preprocessing differs from the original frames")
        ratio = float(np.median(reference / data["depth"][ia]))
        error = float(
            np.median(np.abs(reference - ratio * data["depth"][ia]) / reference)
        )
        if error > 0.08:
            raise ValueError(
                f"Motion bridge {a}-{b} disagrees with anchor depth by {error:.1%}"
            )
        e1, e2 = data["extrinsics"][ia], data["extrinsics"][ib]
        relative = e2[:, :3] @ e1[:, :3].T
        correction = learned[b].T @ relative @ learned[a]
        for frame in learned:
            if frame >= b:
                learned[frame] = learned[frame] @ correction
        local_displacement = -e2[:, :3].T @ e2[:, 3] + e1[:, :3].T @ e1[:, 3]
        result.append(
            {
                "frames": [a, b],
                "owner_window": owner,
                "anchor_scale": ratio,
                "anchor_depth_error": error,
                "local_displacement": local_displacement,
                "local_rotation": e1[:, :3],
                "reference_depth": float(np.median(reference)),
                "checks": checks,
            }
        )
    return result


def bridge_edges(bridges, images, scales):
    edges, checks = [], []
    for bridge in bridges:
        a, b = bridge["frames"]
        if a not in images or b not in images:
            raise ValueError("Bridge endpoints need registered photogrammetry cameras")
        scale = scales[bridge["owner_window"]]
        displacement = (
            images[a]["extrinsics"][:, :3].T
            @ bridge["local_rotation"]
            @ bridge["local_displacement"]
            * bridge["anchor_scale"]
            * scale
        )
        depth = bridge["reference_depth"] * scale
        edges.append((a, b, displacement, 5 / depth**2, depth))
        checks.append(
            {
                k: v
                for k, v in bridge.items()
                if k not in ("local_displacement", "local_rotation")
            }
        )
    return edges, checks
