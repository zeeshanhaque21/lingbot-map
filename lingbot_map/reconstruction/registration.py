"""Recover camera translations from feature tracks and consistent learned depth.

Global monocular SfM can collapse scene scale while retaining small reprojection
errors. Keep its rotations and correspondences, and constrain translations with
depth in one shared scale. This remains relative geometry, not a metric survey.
"""

import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr
from scipy.spatial.transform import Rotation

from .colmap_io import read_model
from .geometry import preprocessing_transform
from .io import digest, write_json, write_npz


def learned_rotations(files, ranges):
    """Align overlapping camera orientations without translation or scale fitting."""
    rotations, checks = {}, []
    for chunk, file in enumerate(files):
        with np.load(file) as data:
            ids, extrinsics = data["frame_ids"], data["extrinsics"]
        shared = [
            (i, int(frame)) for i, frame in enumerate(ids) if int(frame) in rotations
        ]
        transform = np.eye(3)
        if shared:
            candidates = np.stack(
                [rotations[frame].T @ extrinsics[i, :, :3] for i, frame in shared]
            )
            transform = Rotation.from_matrix(candidates).mean().as_matrix()
            residual = Rotation.from_matrix(candidates @ transform.T).magnitude()
            error = float(np.degrees(np.quantile(residual, 0.95)))
            if error > 10:
                raise ValueError(
                    f"Learned orientation overlap exceeds 10 degrees at window {file.stem}"
                )
            checks.append(
                {
                    "window": int(file.stem),
                    "p95_orientation_disagreement_degrees": error,
                }
            )
        for i, frame in enumerate(ids):
            # Use each frame from the earlier window to keep overlapping poses stable.
            rotations.setdefault(int(frame), extrinsics[i, :, :3] @ transform.T)
    return rotations, checks


def stabilize_rotations(images, learned):
    """Align reliable SfM rotation groups to a continuous learned orientation frame."""
    tracks = defaultdict(list)
    for frame, image in images.items():
        for observation in image["observations"]:
            if observation[2] >= 0:
                tracks[int(observation[2])].append(frame)
    pairs = defaultdict(int)
    for frames in tracks.values():
        frames.sort()
        for a, b in pairwise(frames):
            pairs[a, b] += 1
        if len(frames) > 2:
            pairs[frames[0], frames[-1]] += 1
    ids = sorted(images)
    index = {frame: i for i, frame in enumerate(ids)}
    rows, columns = [], []
    for (a, b), count in pairs.items():
        if count < 12:
            continue
        relative = images[b]["extrinsics"][:, :3] @ images[a]["extrinsics"][:, :3].T
        reference = learned[b] @ learned[a].T
        error = np.degrees(Rotation.from_matrix(relative @ reference.T).magnitude())
        if error <= 10:
            rows.extend([index[a], index[b]])
            columns.extend([index[b], index[a]])
    graph = coo_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(len(ids), len(ids))
    ).tocsr()
    count, labels = connected_components(graph, directed=False)
    checks = []
    for group in range(count):
        frames = [frame for frame, label in zip(ids, labels) if label == group]
        candidates = np.stack(
            [learned[frame].T @ images[frame]["extrinsics"][:, :3] for frame in frames]
        )
        transform = Rotation.from_matrix(candidates).mean().as_matrix()
        error = np.degrees(Rotation.from_matrix(candidates @ transform.T).magnitude())
        checks.append(
            {
                "frames": frames,
                "p95_orientation_disagreement_degrees": float(np.quantile(error, 0.95)),
            }
        )
        for frame in frames:
            images[frame] = {
                **images[frame],
                "extrinsics": images[frame]["extrinsics"].copy(),
            }
            images[frame]["extrinsics"][:, :3] = (
                images[frame]["extrinsics"][:, :3] @ transform.T
            )
    return checks


def solve_centers(frame_ids, edges):
    """Robust translation graph; exclude disconnected camera components."""
    index = {frame: i for i, frame in enumerate(frame_ids)}
    rows, columns, values = [], [], []
    for i, (a, b, *_rest) in enumerate(edges):
        rows.extend([i, i])
        columns.extend([index[a], index[b]])
        values.extend([-1, 1])
    if not edges:
        raise ValueError(
            "No camera pairs have sufficient depth-supported feature tracks"
        )
    incidence = coo_matrix(
        (values, (rows, columns)), shape=(len(edges), len(index))
    ).tocsr()
    _, labels = connected_components(incidence.T @ incidence, directed=False)
    largest = int(np.argmax(np.bincount(labels)))
    connected = labels == largest
    if connected.mean() < 0.9:
        raise ValueError(
            f"Depth translation graph connects only {connected.sum()}/{len(index)} cameras; capture more overlapping views"
        )
    anchor = int(np.flatnonzero(connected)[0])
    rows.append(len(edges))
    columns.append(anchor)
    values.append(1)
    matrix = coo_matrix(
        (values, (rows, columns)), shape=(len(edges) + 1, len(index))
    ).tocsr()
    rhs = np.vstack([e[2] for e in edges] + [np.zeros(3)])
    base = np.array([e[3] for e in edges] + [1e5])
    depths = np.array([e[4] for e in edges])
    weights = base.copy()
    for _ in range(5):
        weighted = matrix.multiply(np.sqrt(weights)[:, None]).tocsr()
        centers = np.column_stack(
            [
                lsqr(
                    weighted,
                    rhs[:, j] * np.sqrt(weights),
                    atol=1e-9,
                    btol=1e-9,
                    iter_lim=5000,
                )[0]
                for j in range(3)
            ]
        )
        error = np.linalg.norm(matrix @ centers - rhs, axis=1)
        weights[:-1] = base[:-1] * np.minimum(
            1, 0.02 * depths / np.maximum(error[:-1], 1e-8)
        )
    return centers, connected, np.quantile(error[:-1] / depths, [0.5, 0.9, 0.95])


def motion_constraint(data, i, j, images, scale):
    """Check a learned motion against depth after substituting global rotations."""
    a, b = int(data["frame_ids"][i]), int(data["frame_ids"][j])
    e1, e2 = data["extrinsics"][i], data["extrinsics"][j]
    r1, r2 = images[a]["extrinsics"][:, :3], images[b]["extrinsics"][:, :3]
    center1, center2 = -e1[:, :3].T @ e1[:, 3], -e2[:, :3].T @ e2[:, 3]
    displacement = (
        ((r1.T @ e1[:, :3] + r2.T @ e2[:, :3]) / 2) @ (center2 - center1) * scale
    )
    rotation_difference = (r2 @ r1.T) @ (e2[:, :3] @ e1[:, :3].T).T
    angle = float(
        np.degrees(np.arccos(np.clip((np.trace(rotation_difference) - 1) / 2, -1, 1)))
    )
    h, w = data["depth"][i].shape
    intrinsics = []
    for frame in (a, b):
        camera = images[frame]["camera"]
        intrinsics.append(
            preprocessing_transform(camera["width"], camera["height"], w, h)
            @ camera["intrinsics"]
        )
    y, x = np.mgrid[0:h:8, 0:w:8]
    depth = data["depth"][i, ::8, ::8] * scale
    rays = np.stack([x, y, np.ones_like(x)], axis=-1) @ np.linalg.inv(intrinsics[0]).T
    points = (rays * depth[..., None]) @ r1
    camera2 = (points - displacement) @ r2.T
    projected = camera2 @ intrinsics[1].T
    uv = projected[..., :2] / np.maximum(projected[..., 2:], 1e-8)
    target = cv2.remap(
        data["depth"][j] * scale,
        uv[..., 0].astype(np.float32),
        uv[..., 1].astype(np.float32),
        cv2.INTER_LINEAR,
    )
    valid = (target > 0) & (camera2[..., 2] > 0) & (depth > 0)
    error = np.abs(target - camera2[..., 2]) / np.maximum(target, 1e-8)
    fraction = float((valid & (error < 0.05)).sum() / max(valid.sum(), 1))
    accepted = angle < 10 and valid.sum() >= 100 and fraction >= 0.25
    stats = {
        "frames": [a, b],
        "relative_rotation_error_degrees": angle,
        "overlap_pixels": int(valid.sum()),
        "depth_support_fraction": fraction,
        "accepted": bool(accepted),
    }
    reference_depth = float(np.median(depth))
    edge = (
        (a, b, displacement, 5 * fraction / reference_depth**2, reference_depth)
        if accepted
        else None
    )
    return edge, stats


def collect_tracks(source, images, files, ranges):
    tracks, scales, frame_depths, overlaps = defaultdict(list), [], {}, []
    motion_edges, motion_checks = [], []
    previous, scale = None, 1.0
    for chunk, file in enumerate(files):
        data = dict(np.load(file))
        if previous is not None:
            _, ii, jj = np.intersect1d(
                previous["frame_ids"], data["frame_ids"], return_indices=True
            )
            if not len(ii):
                raise ValueError("Depth windows do not overlap")
            x, y = previous["depth"][ii, ::8, ::8], data["depth"][jj, ::8, ::8]
            keep = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
            ratio = float(np.median(x[keep] / y[keep]))
            error = float(np.median(np.abs(x[keep] - y[keep] * ratio) / x[keep]))
            if not np.isfinite(ratio) or error > 0.1:
                raise ValueError(
                    f"Overlapping depth disagrees by {error:.1%}; inspect window {file.stem}"
                )
            scale *= ratio
            overlaps.append(
                {
                    "window": int(file.stem),
                    "scale_ratio": ratio,
                    "median_relative_depth_error": error,
                }
            )
        scales.append(float(scale))
        start, end = ranges[chunk]
        low = start if chunk == 0 else (start + ranges[chunk - 1][1]) // 2
        high = end if chunk == len(files) - 1 else (end + ranges[chunk + 1][0]) // 2
        for i, frame in enumerate(data["frame_ids"]):
            frame = int(frame)
            if not low <= frame < high or frame not in images:
                continue
            image = images[frame]
            for j in range(i + 1, min(i + 4, len(data["frame_ids"]))):
                if int(data["frame_ids"][j]) in images:
                    edge, stats = motion_constraint(data, i, j, images, scale)
                    motion_checks.append(stats)
                    if edge is not None:
                        motion_edges.append(edge)
                    break
            r = image["extrinsics"][:, :3]
            camera = image["camera"]
            depth = data["depth"][i] * scale
            h, w = depth.shape
            frame_depths[frame] = float(np.median(depth))
            observations = image["observations"]
            observations = observations[observations[:, 2] >= 0]
            affine = preprocessing_transform(camera["width"], camera["height"], w, h)
            uv = (
                np.column_stack([observations[:, :2], np.ones(len(observations))])
                @ affine.T
            )
            u, v = uv[:, 0].astype(np.float32), uv[:, 1].astype(np.float32)
            z = cv2.remap(depth, u[:, None], v[:, None], cv2.INTER_LINEAR).ravel()
            confidence = cv2.remap(
                data["confidence"][i], u[:, None], v[:, None], cv2.INTER_LINEAR
            ).ravel()
            good = (
                (z > 0)
                & np.isfinite(z)
                & (confidence >= np.quantile(data["confidence"][i], 0.25))
            )
            rays = (
                np.column_stack([observations[:, :2], np.ones(len(observations))])
                @ np.linalg.inv(camera["intrinsics"]).T
            )
            offsets = (rays * z[:, None]) @ r
            for observation, offset in zip(observations[good], offsets[good]):
                tracks[int(observation[2])].append((frame, offset, observation[:2]))
        previous = {k: data[k] for k in ("frame_ids", "depth")}
    return tracks, scales, frame_depths, overlaps, motion_edges, motion_checks


def track_edges(tracks, frame_depths):
    pairs = defaultdict(list)
    for track, observations in tracks.items():
        if track % 5 == 0:
            continue
        observations.sort(key=lambda x: x[0])
        links = list(pairwise(observations))
        if len(observations) > 2:
            links.append((observations[0], observations[-1]))
        for a, b in links:
            pairs[a[0], b[0]].append(a[1] - b[1])
    edges = []
    for (a, b), samples in pairs.items():
        if len(samples) < 12:
            continue
        samples = np.array(samples)
        center = np.median(samples, axis=0)
        residual = np.linalg.norm(samples - center, axis=1)
        good = residual < max(np.median(residual) * 3, 1e-5)
        displacement = np.mean(samples[good], axis=0)
        depth = (frame_depths[a] + frame_depths[b]) / 2
        edges.append((a, b, displacement, min(int(good.sum()), 100) / depth**2, depth))
    return edges


def evaluate_tracks(tracks, images, centers):
    errors = []
    for track, observations in tracks.items():
        observations = [o for o in observations if o[0] in centers]
        if track % 5 or len(observations) < 3:
            continue
        world = np.array([centers[f] + p for f, p, _ in observations])
        for i, (frame, _, observed) in enumerate(observations):
            xyz = np.median(np.delete(world, i, axis=0), axis=0)
            image = images[frame]
            camera = image["extrinsics"][:, :3] @ (xyz - centers[frame])
            if camera[2] > 0:
                projected = image["camera"]["intrinsics"] @ camera
                errors.append(
                    float(np.linalg.norm(projected[:2] / projected[2] - observed))
                )
    if not errors:
        raise ValueError("No withheld feature tracks available to check registration")
    return {
        "observations": len(errors),
        "pixel_error_p50_p90_p95": np.quantile(errors, [0.5, 0.9, 0.95]).tolist(),
        "interpretation": "Tracks withheld from translation fitting; rotations and depths still share source images. Not independent ground truth.",
    }


def register(source, destination, colmap_model, bridges=None):
    source, destination, colmap_model = (
        Path(source),
        Path(destination),
        Path(colmap_model),
    )
    manifest = json.loads((source / "input.json").read_text())
    config = json.loads((source / "inference.json").read_text())
    if config.get("stored_pose_convention") != "world-to-camera":
        raise ValueError("Normalize legacy pose archives before camera registration")
    from .inference import window_ranges

    ranges = list(
        window_ranges(len(manifest["frames"]), config["window"], config["overlap"])
    )
    files = sorted((source / "windows").glob("*.npz"))
    if [int(f.stem) for f in files] != [r[0] for r in ranges]:
        raise ValueError("Finish source inference before registration")
    signature = {
        "method_version": 4,
        "bridges_sha256": digest(Path(bridges)) if bridges else None,
        "source_inference_sha256": digest(source / "inference.json"),
        "source_windows": {p.name: digest(p) for p in files},
        "colmap_files": {
            name: digest(colmap_model / name)
            for name in ("cameras.txt", "images.txt", "points3D.txt")
        },
    }
    if destination.exists():
        marker = destination / "registration-input.json"
        if not marker.exists() or json.loads(marker.read_text()) != signature:
            raise ValueError(
                "Registration output belongs to different inputs; use a new directory"
            )
        completed = destination / "registration.json"
        if completed.exists():
            saved = json.loads(completed.read_text())
            if all(
                (destination / name).exists() and digest(destination / name) == checksum
                for name, checksum in saved["files"].items()
            ):
                return
            raise ValueError(
                "Completed registration archives changed; preserve them and use a new directory"
            )
    else:
        destination.mkdir(parents=True)
        write_json(destination / "registration-input.json", signature)
    image_names, _ = read_model(colmap_model)
    images = {
        frame: image_names[Path(item["file"]).name]
        for frame, item in enumerate(manifest["frames"])
        if Path(item["file"]).name in image_names
    }
    learned, orientation_overlaps = learned_rotations(files, ranges)
    bridge_data = []
    if bridges:
        from .bridges import load_bridges

        bridge_data = load_bridges(bridges, source, files, ranges, learned)
    orientation_groups = stabilize_rotations(images, learned)
    tracks, scales, frame_depths, overlaps, motion_edges, motion_checks = (
        collect_tracks(source, images, files, ranges)
    )
    edges = track_edges(tracks, frame_depths) + motion_edges
    from .bridges import bridge_edges

    extra_edges, bridge_checks = bridge_edges(bridge_data, images, scales)
    edges.extend(extra_edges)
    write_json(
        destination / "registration-diagnostics.json",
        {
            "edges": len(edges),
            "motion_edges": len(motion_edges),
            "motion_checks": motion_checks,
            "depth_overlaps": overlaps,
            "orientation_groups": orientation_groups,
            "orientation_overlaps": orientation_overlaps,
            "dense_bridges": bridge_checks,
            "edge_frames": [[e[0], e[1]] for e in edges],
        },
    )
    ids = sorted(images)
    positions, connected, residual = solve_centers(ids, edges)
    centers = {
        frame: center for frame, center, keep in zip(ids, positions, connected) if keep
    }
    evaluation = evaluate_tracks(tracks, images, centers)
    # Coarse rejection gate scaled to original image resolution.
    width = next(iter(images.values()))["camera"]["width"]
    if evaluation["pixel_error_p50_p90_p95"][0] > width * 0.01:
        raise ValueError(
            "Depth-constrained camera registration exceeds the reprojection screening threshold"
        )
    config.update(
        poses_global=True,
        pose_source="Depth and motion constrained pose graph with aligned SfM rotations",
        registration_signature=signature,
    )
    frames_link = destination / "frames"
    if not frames_link.exists():
        frames_link.symlink_to((source / "frames").resolve(), target_is_directory=True)
    write_json(destination / "input.json", manifest)
    write_json(destination / "inference.json", config)
    for chunk, file in enumerate(files):
        target = destination / "windows" / file.name
        if target.exists():
            continue  # Each archive is atomically published for this exact signature.
        data = dict(np.load(file))
        data["depth"] *= scales[chunk]
        registered = np.zeros(len(data["frame_ids"]), bool)
        for i, frame in enumerate(data["frame_ids"]):
            if int(frame) not in centers:
                continue
            item = images[int(frame)]
            r = item["extrinsics"][:, :3]
            data["extrinsics"][i] = np.column_stack([r, -r @ centers[int(frame)]])
            h, w = data["depth"][i].shape
            camera = item["camera"]
            data["intrinsics"][i] = (
                preprocessing_transform(camera["width"], camera["height"], w, h)
                @ camera["intrinsics"]
            )
            registered[i] = True
        data["registered"] = registered
        write_npz(target, **data)
        print(f"Registered window {chunk + 1}/{len(files)}", flush=True)
    write_json(
        destination / "registration.json",
        {
            "method": config["pose_source"],
            "registered_frames": len(centers),
            "input_frames": len(manifest["frames"]),
            "edges": len(edges),
            "learned_motion_edges": len(motion_edges),
            "motion_checks": motion_checks,
            "dense_bridges": bridge_checks,
            "orientation_groups": orientation_groups,
            "orientation_overlaps": orientation_overlaps,
            "depth_scales": scales,
            "depth_overlap_checks": overlaps,
            "relative_graph_residual_p50_p90_p95": residual.tolist(),
            "withheld_track_check": evaluation,
            "metric_scale_known": False,
            "files": {
                str(p.relative_to(destination)): digest(p)
                for p in (destination / "windows").glob("*.npz")
            },
        },
    )
