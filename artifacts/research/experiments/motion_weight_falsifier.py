"""Hold images/rotations/depth fixed; vary native-motion weight in the real graph."""

import argparse
import json
from pathlib import Path

import numpy as np

from lingbot_map.reconstruction.bridges import bridge_edges, load_bridges
from lingbot_map.reconstruction.colmap_io import read_model
from lingbot_map.reconstruction.geometry import preprocessing_transform
from lingbot_map.reconstruction.inference import window_ranges
from lingbot_map.reconstruction.registration import (
    collect_tracks,
    evaluate_tracks,
    learned_rotations,
    select_learned_cameras,
    solve_centers,
    track_edges,
)

from .local_pose_falsifier import reconstruct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--colmap-model", type=Path, required=True)
    parser.add_argument("--bridges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / "results.jsonl"
    if log.exists():
        raise ValueError("Choose a new output directory to preserve the previous comparison")
    manifest = json.loads((args.source / "input.json").read_text())
    configuration = json.loads((args.source / "inference.json").read_text())
    ranges = list(window_ranges(len(manifest["frames"]), configuration["window"], configuration["overlap"]))
    files = sorted((args.source / "windows").glob("*.npz"))
    names, _ = read_model(args.colmap_model)
    images = {frame: names[Path(item["file"]).name] for frame, item in enumerate(manifest["frames"]) if Path(item["file"]).name in names}
    learned, _ = learned_rotations(files, ranges)
    bridges = load_bridges(args.bridges, args.source, files, ranges, learned)
    select_learned_cameras(images, learned, files, ranges)
    tracks, scales, frame_depths, _, motion, _ = collect_tracks(args.source, images, files, ranges)
    feature_edges = track_edges(tracks, frame_depths)
    extra, _ = bridge_edges(bridges, images, scales)
    ids = sorted(images)
    for factor in [1, 10, 100, 1000]:
        edges = feature_edges + [(a, b, delta, weight * factor, depth) for a, b, delta, weight, depth in motion + extra]
        positions, connected, residual = solve_centers(ids, edges)
        centers = {frame: position for frame, position, keep in zip(ids, positions, connected) if keep}
        track_check = evaluate_tracks(tracks, images, centers)
        print(json.dumps({"factor": factor, "track_check": track_check, "graph_residual": residual.tolist()}), flush=True)
        for start, target in [(936, 945), (504, 515)]:
            chunk = next(i for i, p in enumerate(files) if int(p.stem) == start)
            data = dict(np.load(files[chunk]))
            data["depth"] *= scales[chunk]
            for i, frame in enumerate(data["frame_ids"]):
                r = images[int(frame)]["extrinsics"][:, :3]
                data["extrinsics"][i] = np.column_stack([r, -r @ centers[int(frame)]])
                camera = images[int(frame)]["camera"]
                h, w = data["depth"][i].shape
                data["intrinsics"][i] = preprocessing_transform(camera["width"], camera["height"], w, h) @ camera["intrinsics"]
            result = reconstruct(data, target, 9999, args.output, f"motion-{factor}")
            result["track_check"] = track_check
            with log.open("a") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
