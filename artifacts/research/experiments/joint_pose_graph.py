"""Fit a complete SE3 camera graph and audit its reconstructed mesh.

Uses robust depth-backed feature correspondences, weak existing-motion priors,
and Open3D global optimization. One fifth of tracks remain reserved from fitting.
Completed observations, edges and poses are saved before mesh reconstruction.
"""

import argparse
import json
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from lingbot_map.reconstruction.colmap_io import read_model
from lingbot_map.reconstruction.io import digest, write_json, write_npz

from .pair_pose_falsifier import camera_points, robust_fit


def observations(source, colmap_model, output, cameras):
    saved = output / "observations.npz"
    if saved.exists():
        return dict(np.load(saved))
    images, _ = read_model(colmap_model)
    manifest = json.loads((source / "input.json").read_text())
    frames, tracks, xyz, pixels = [], [], [], []
    for window, path in enumerate(sorted((source / "windows").glob("*.npz"))):
        data = dict(np.load(path))
        for camera in cameras:
            if camera["window"] != window:
                continue
            frame = camera["frame"]
            name = Path(manifest["frames"][frame]["file"]).name
            if name not in images:
                continue
            index = int(np.flatnonzero(data["frame_ids"] == frame)[0])
            for track, (point, pixel) in camera_points(
                data, index, images[name]
            ).items():
                frames.append(frame)
                tracks.append(track)
                xyz.append(point)
                pixels.append(pixel)
        print(f"observed_windows: {window + 1}", flush=True)
    result = {
        "frame": np.asarray(frames, np.int32),
        "track": np.asarray(tracks, np.int64),
        "xyz": np.asarray(xyz, np.float64),
        "pixel": np.asarray(pixels, np.float64),
    }
    write_npz(saved, **result)
    return result


def pair_groups(data, reserved=False):
    selected = np.flatnonzero((data["track"] % 5 == 0) == reserved)
    order = selected[np.lexsort((data["frame"][selected], data["track"][selected]))]
    breaks = np.flatnonzero(np.diff(data["track"][order])) + 1
    groups = defaultdict(list)
    for indices in np.split(order, breaks):
        links = list(pairwise(indices))
        if len(indices) > 2:
            links.append((indices[0], indices[-1]))
        for left, right in links:
            a, b = int(data["frame"][left]), int(data["frame"][right])
            if a != b:
                groups[a, b].append((int(left), int(right)))
    return groups


def information_matrix(points, sigma):
    jacobian = np.zeros((len(points), 3, 6))
    x, y, z = points.T
    jacobian[:, 0, 1], jacobian[:, 0, 2] = z, -y
    jacobian[:, 1, 0], jacobian[:, 1, 2] = -z, x
    jacobian[:, 2, 0], jacobian[:, 2, 1] = y, -x
    jacobian[:, :, 3:] = np.eye(3)
    return (
        np.einsum("nki,nkj->ij", jacobian, jacobian)
        * min(len(points), 100)
        / len(points)
        / sigma**2
    )


def fit_edges(data, cameras, output):
    pairs = pair_groups(data)
    intrinsics = {c["frame"]: np.asarray(c["intrinsics"]) for c in cameras}
    path = output / "edges.jsonl"
    records = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    done = {tuple(r["frames"]) for r in records}
    started = time.monotonic()
    for (a, b), indices in sorted(pairs.items()):
        if (a, b) in done or len(indices) < 12:
            continue
        indices = np.asarray(indices)
        x, y = data["xyz"][indices[:, 0]], data["xyz"][indices[:, 1]]
        _, rotation, translation = robust_fit(x, y)
        projected_points = x @ rotation.T + translation
        error = np.linalg.norm(projected_points - y, axis=1)
        depth = float(np.median(y[:, 2]))
        projected = projected_points @ intrinsics[b].T
        pixel = projected[:, :2] / np.maximum(projected[:, 2:], 1e-8)
        pixel_error = np.linalg.norm(pixel - data["pixel"][indices[:, 1]], axis=1)
        good = (error < depth * 0.05) & (pixel_error < 5) & (projected_points[:, 2] > 0)
        spread = np.linalg.svd(x - x.mean(0), compute_uv=False)
        accepted = (
            good.sum() >= 12
            and np.median(error) < depth * 0.05
            and np.median(pixel_error) < 3
            and spread[1] > spread[0] * 1e-3
        )
        record = {
            "frames": [a, b],
            "tracks": len(indices),
            "inliers": int(good.sum()),
            "median_relative_3d_error": float(np.median(error) / depth),
            "median_pixel_error": float(np.median(pixel_error)),
            "accepted": bool(accepted),
        }
        if accepted:
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = rotation, translation
            sigma = max(float(np.median(error[good])), depth * 0.01)
            record.update(
                transform=transform.tolist(),
                information=information_matrix(y[good], sigma).tolist(),
            )
        with path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
        records.append(record)
        if len(records) % 100 == 0:
            print(
                f"fitted_pairs: {len(records)}\nseconds: {time.monotonic() - started:.1f}",
                flush=True,
            )
    return records


def evaluate(data, cameras, poses):
    frames = [c["frame"] for c in cameras]
    index = {frame: i for i, frame in enumerate(frames)}
    intrinsics = {c["frame"]: np.asarray(c["intrinsics"]) for c in cameras}
    pairs = pair_groups(data, reserved=True)
    errors, relative, by_frame = [], [], defaultdict(list)
    for (a, b), indices in pairs.items():
        rows = np.asarray(indices)
        transform = np.linalg.inv(poses[index[b]]) @ poses[index[a]]
        points = data["xyz"][rows[:, 0]] @ transform[:3, :3].T + transform[:3, 3]
        projected = points @ intrinsics[b].T
        uv = projected[:, :2] / np.maximum(projected[:, 2:], 1e-8)
        error = np.linalg.norm(uv - data["pixel"][rows[:, 1]], axis=1)
        errors.extend(error.tolist())
        relative.extend(
            (
                np.linalg.norm(points - data["xyz"][rows[:, 1]], axis=1)
                / data["xyz"][rows[:, 1], 2]
            ).tolist()
        )
        by_frame[b].extend(error.tolist())
    return {
        "reserved_observations": len(errors),
        "pixel_error_p50_p90_p95": np.quantile(errors, [0.5, 0.9, 0.95]).tolist(),
        "relative_3d_error_p50_p90_p95": np.quantile(
            relative, [0.5, 0.9, 0.95]
        ).tolist(),
        "per_frame": {
            str(frame): {
                "observations": len(values),
                "median_pixel_error": float(np.median(values)),
            }
            for frame, values in by_frame.items()
        },
        "interpretation": "Tracks reserved from pair fitting; shared depth and source images, not independent survey accuracy",
    }


def optimize(data, cameras, edges, output, prior_weight):
    saved = output / "optimized-poses.npz"
    if saved.exists():
        return np.load(saved)["poses"]
    initial = np.asarray([c["camera_to_world"] for c in cameras])
    index = {c["frame"]: i for i, c in enumerate(cameras)}
    registration = o3d.pipelines.registration
    graph = registration.PoseGraph()
    for pose in initial:
        graph.nodes.append(registration.PoseGraphNode(pose))
    depth = float(np.median(data["xyz"][:, 2]))
    prior = np.diag([depth**2] * 3 + [1.0] * 3) * (
        100 * prior_weight / (0.02 * depth) ** 2
    )
    for i in range(len(cameras) - 1):
        transform = np.linalg.inv(initial[i + 1]) @ initial[i]
        graph.edges.append(
            registration.PoseGraphEdge(i, i + 1, transform, prior, uncertain=False)
        )
    accepted = [e for e in edges if e["accepted"]]
    for edge in accepted:
        a, b = edge["frames"]
        graph.edges.append(
            registration.PoseGraphEdge(
                index[a],
                index[b],
                np.asarray(edge["transform"]),
                np.asarray(edge["information"]),
                uncertain=abs(a - b) > 3,
            )
        )
    o3d.io.write_pose_graph(str(output / "initial-graph.json"), graph)
    print(
        f"camera_nodes: {len(cameras)}\nfitted_edges: {len(accepted)}\nprior_edges: {len(cameras) - 1}",
        flush=True,
    )
    criteria = registration.GlobalOptimizationConvergenceCriteria()
    criteria.max_iteration = 100
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        registration.global_optimization(
            graph,
            registration.GlobalOptimizationLevenbergMarquardt(),
            criteria,
            registration.GlobalOptimizationOption(
                max_correspondence_distance=0.05 * depth,
                edge_prune_threshold=0.25,
                preference_loop_closure=1.0,
                reference_node=0,
            ),
        )
    poses = np.asarray([node.pose for node in graph.nodes])
    if not np.isfinite(poses).all():
        raise ValueError("Global optimization produced non-finite poses")
    o3d.io.write_pose_graph(str(output / "optimized-graph.json"), graph)
    movement = np.linalg.norm(poses[:, :3, 3] - initial[:, :3, 3], axis=1) / depth
    rotation = (
        Rotation.from_matrix(
            poses[:, :3, :3] @ initial[:, :3, :3].transpose(0, 2, 1)
        ).magnitude()
        * 180
        / np.pi
    )
    write_json(
        output / "camera-comparison.json",
        {
            "baseline": evaluate(data, cameras, initial),
            "optimized": evaluate(data, cameras, poses),
            "relative_translation_change_p50_p90_p95": np.quantile(
                movement, [0.5, 0.9, 0.95]
            ).tolist(),
            "rotation_change_degrees_p50_p90_p95": np.quantile(
                rotation, [0.5, 0.9, 0.95]
            ).tolist(),
            "accepted_feature_edges": len(accepted),
            "retained_graph_edges": len(graph.edges),
            "weak_prior_edges": len(cameras) - 1,
        },
    )
    write_npz(saved, poses=poses)
    return poses


def publish(
    source,
    output,
    cameras,
    poses,
    signature,
    depth_scales=None,
    pose_source="Joint SE3 camera graph from depth-backed image features",
):
    candidate = output / "candidate"
    candidate.mkdir(exist_ok=True)
    for name in ["input.json", "frames"]:
        destination = candidate / name
        if not destination.exists():
            destination.symlink_to((source / name).resolve())
    config = json.loads((source / "inference.json").read_text())
    config.update(
        pose_source=pose_source,
        joint_pose_signature=signature,
    )
    write_json(candidate / "inference.json", config)
    extrinsics = {
        camera["frame"]: np.linalg.inv(pose)[:3] for camera, pose in zip(cameras, poses)
    }
    scales = (
        {camera["frame"]: scale for camera, scale in zip(cameras, depth_scales)}
        if depth_scales is not None
        else {}
    )
    for path in sorted((source / "windows").glob("*.npz")):
        destination = candidate / "windows" / path.name
        if destination.exists():
            continue
        data = dict(np.load(path))
        for i, frame in enumerate(data["frame_ids"]):
            if int(frame) in extrinsics:
                data["extrinsics"][i] = extrinsics[int(frame)]
            if int(frame) in scales:
                data["depth"][i] *= scales[int(frame)]
        write_npz(destination, **data)
        print(f"published_window: {path.stem}", flush=True)
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--colmap-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-weight", type=float, default=0.05)
    args = parser.parse_args()
    if not np.isfinite(args.prior_weight) or args.prior_weight <= 0:
        parser.error("--prior-weight must be finite and positive")
    source_config = json.loads((args.source / "inference.json").read_text())
    if not source_config.get("poses_global"):
        parser.error(
            "--source must contain registered cameras in shared world coordinates"
        )
    signature = {
        "version": 1,
        "prior_weight": args.prior_weight,
        "source_inference": digest(args.source / "inference.json"),
        "source_cameras": digest(args.source / "model/cameras.json"),
        "source_windows": {
            p.name: digest(p) for p in sorted((args.source / "windows").glob("*.npz"))
        },
        "colmap": {
            name: digest(args.colmap_model / name)
            for name in ["images.txt", "cameras.txt", "points3D.txt"]
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "input.json"
    if marker.exists() and json.loads(marker.read_text()) != signature:
        raise ValueError("Inputs changed; choose a new --output")
    write_json(marker, signature)
    cameras = json.loads((args.source / "model/cameras.json").read_text())
    data = observations(args.source, args.colmap_model, args.output, cameras)
    edges = fit_edges(data, cameras, args.output)
    poses = optimize(data, cameras, edges, args.output, args.prior_weight)
    candidate = publish(args.source, args.output, cameras, poses, signature)
    from lingbot_map.reconstruction.fusion import fuse
    from lingbot_map.reconstruction.validation import validate

    fuse(candidate)
    results = {}
    for name, asset in [("full", None), ("app", "property.glb")]:
        report = validate(candidate, asset_name=asset)
        results[name] = {k: v for k, v in report.items() if k != "views"}
        results[name]["failed_views"] = [
            v["frame"] for v in report["views"] if v["supported_depth_fraction"] < 0.4
        ]
        write_json(args.output / "results.json", results)
        print(
            f"asset: {name}\nview_consistency_gate: {str(report['view_consistency_gate']).lower()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
