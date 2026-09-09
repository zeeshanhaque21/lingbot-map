"""Test MapAnything depth with fixed registered cameras, keeping failed outputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from artifacts.research.experiments.local_pose_falsifier import reconstruct
from artifacts.research.experiments.mapanything_mac import digest, processed_sources
from lingbot_map.reconstruction.fusion import fuse
from lingbot_map.reconstruction.geometry import robust_similarity
from lingbot_map.reconstruction.validation import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing output; choose a new directory")
    manifest = json.loads((args.predictions / "input.json").read_text())
    inference = json.loads((args.predictions / "inference.json").read_text())
    selected = [
        {**frame, "id": frame["original_frame"]} for frame in manifest["frames"]
    ]
    if len(selected) < 8:
        parser.error("At least eight frames are required for fusion")
    if any((frame["id"] - index) % 10 for index, frame in enumerate(selected)):
        parser.error("Frame offsets must preserve the reserved-view split")
    paths, hashes = processed_sources(args.cameras, selected)
    with np.load(args.predictions / "windows/000000.npz") as archive:
        data = dict(archive)
    source_extrinsics, source_intrinsics = [], []
    previous, window = None, None
    for path, frame in zip(paths, selected):
        if path != previous:
            with np.load(path) as archive:
                window = dict(archive)
            previous = path
        index = int(np.flatnonzero(window["frame_ids"] == frame["id"])[0])
        source_extrinsics.append(window["extrinsics"][index])
        source_intrinsics.append(window["intrinsics"][index])
    source_extrinsics = np.stack(source_extrinsics)
    source_intrinsics = np.stack(source_intrinsics)
    source_centers = -np.einsum(
        "nji,nj->ni", source_extrinsics[:, :, :3], source_extrinsics[:, :, 3]
    )
    predicted_centers = -np.einsum(
        "nji,nj->ni", data["extrinsics"][:, :, :3], data["extrinsics"][:, :, 3]
    )
    # Reserve exactly the cameras that will be withheld from fusion.
    training = data["frame_ids"] % 10 != 5
    transform, residual = robust_similarity(
        predicted_centers[training], source_centers[training]
    )
    scale = float(np.cbrt(np.linalg.det(transform[:3, :3])))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid camera-trajectory scale")
    predicted_aligned = predicted_centers @ transform[:3, :3].T + transform[:3, 3]
    depth_unit = float(np.median(data["depth"]) * scale)
    report = {
        "source_prediction_sha256": digest(args.predictions / "windows/000000.npz"),
        "camera_window_sha256": hashes,
        "training_cameras": int(training.sum()),
        "reserved_cameras": int((~training).sum()),
        "depth_scale": scale,
        "camera_scale_fit": transform.tolist(),
        "training_center_residual_p50_p95_relative_to_depth": (
            np.quantile(residual, [0.5, 0.95]) / depth_unit
        ).tolist(),
        "reserved_center_residual_p50_p95_relative_to_depth": (
            np.quantile(
                np.linalg.norm(
                    predicted_aligned[~training] - source_centers[~training], axis=1
                ),
                [0.5, 0.95],
            )
            / depth_unit
        ).tolist(),
        "interpretation": "Single depth scale estimated from training camera centers; all output K and poses fixed to the registered source. No surveyed accuracy claim.",
    }
    data["depth"] = data["depth"] * scale
    data["extrinsics"] = source_extrinsics
    data["intrinsics"] = source_intrinsics
    args.output.mkdir(parents=True)
    (args.output / "windows").mkdir()
    np.savez_compressed(args.output / "windows/000000.npz", **data)
    inference.update(
        {
            "window": len(selected),
            "overlap": min(24, len(selected) // 2),
            "poses_global": True,
            "pose_source": "MapAnything depth with fixed registered cameras",
            "fixed_camera_experiment": report,
        }
    )
    for name, value in [
        ("input.json", manifest),
        ("inference.json", inference),
        ("anchoring.json", report),
    ]:
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    # Compare the same original viewpoints and retain both neighborhoods.
    data["frame_ids"] = np.asarray([frame["id"] for frame in selected])
    with (args.output / "local-comparison.jsonl").open("x") as stream:
        for target in [775, 785, 795]:
            if target not in data["frame_ids"]:
                continue
            for radius, label in [(100, "fixed-window"), (8, "fixed-neighbors")]:
                result = reconstruct(data, target, radius, args.output, label)
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                print(json.dumps(result), flush=True)
    fuse(args.output)
    results = {}
    for name, asset in [("full", None), ("app", "property.glb")]:
        result = validate(args.output, asset_name=asset)
        results[name] = {k: v for k, v in result.items() if k != "views"}
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
