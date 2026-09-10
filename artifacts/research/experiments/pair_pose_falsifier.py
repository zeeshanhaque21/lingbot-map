"""Compare camera-pair fits on feature tracks reserved from fitting.

Run with --native <raw run> --registered <global run> --colmap-model <text model>
and --output <new directory>. Retains the unchanged registered and native poses
beside robust rigid and similarity fits. These checks are not surveyed accuracy.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lingbot_map.reconstruction.colmap_io import read_model


def fit(x, y, similarity=False):
    center_x, center_y = x.mean(0), y.mean(0)
    a, b = x - center_x, y - center_y
    u, singular, vt = np.linalg.svd(a.T @ b)
    parity = np.ones(3)
    parity[-1] = np.linalg.det(vt.T @ u.T)
    rotation = (vt.T * parity) @ u.T
    scale = float((singular * parity).sum() / np.square(a).sum()) if similarity else 1.0
    return scale, rotation, center_y - scale * rotation @ center_x


def robust_fit(x, y, similarity=False):
    rng = np.random.default_rng(20260909)
    best, score = None, float("inf")
    for _ in range(80):
        sample = rng.choice(len(x), 6, replace=False)
        candidate = fit(x[sample], y[sample], similarity)
        scale, rotation, translation = candidate
        residual = np.linalg.norm(scale * x @ rotation.T + translation - y, axis=1)
        if np.median(residual) < score:
            best, score = candidate, float(np.median(residual))
    for _ in range(5):
        scale, rotation, translation = best
        residual = np.linalg.norm(scale * x @ rotation.T + translation - y, axis=1)
        keep = residual < max(float(np.median(residual)) * 2.5, float(np.median(y[:, 2])) * 0.01)
        if keep.sum() < 6:
            break
        best = fit(x[keep], y[keep], similarity)
    return best


def camera_points(data, index, image):
    observations = image["observations"]
    observations = observations[observations[:, 2] >= 0]
    height, width = data["depth"][index].shape
    original = image["camera"]
    resized_height = round(original["height"] * width / original["width"] / 14) * 14
    uv = observations[:, :2] * [width / original["width"], resized_height / original["height"]]
    uv[:, 1] -= max(0, (resized_height - height) // 2)
    depth = cv2.remap(data["depth"][index], uv[:, 0, None].astype(np.float32), uv[:, 1, None].astype(np.float32), cv2.INTER_LINEAR).ravel()
    confidence = cv2.remap(data["confidence"][index], uv[:, 0, None].astype(np.float32), uv[:, 1, None].astype(np.float32), cv2.INTER_LINEAR).ravel()
    valid = (depth > 0) & np.isfinite(depth) & (confidence >= np.quantile(data["confidence"][index], 0.25))
    points = (np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(data["intrinsics"][index]).T) * depth[:, None]
    return {int(track): (point, pixel) for track, point, pixel in zip(observations[valid, 2], points[valid], uv[valid])}


def compare(native, registered, colmap_model, output):
    output.mkdir(parents=True, exist_ok=True)
    log = output / "results.jsonl"
    if log.exists():
        raise ValueError("Choose a new --output to preserve the comparison")
    images, _ = read_model(colmap_model)
    pairs = [(288, 355), (504, 515), (504, 535), (648, 665), (720, 755), (720, 785), (720, 795), (936, 945)]
    for start, target in pairs:
        raw = dict(np.load(native / "windows" / f"{start:06d}.npz"))
        data = dict(np.load(registered / "windows" / f"{start:06d}.npz"))
        a, b = target - 1, target + 1
        i, j = [int(np.flatnonzero(data["frame_ids"] == frame)[0]) for frame in [a, b]]
        points = [camera_points(data, index, images[f"{frame:06d}.jpg"]) for index, frame in [(i, a), (j, b)]]
        tracks = sorted(set(points[0]) & set(points[1]))
        held = np.array([track % 5 == 0 for track in tracks])
        if held.sum() < 4 or (~held).sum() < 12:
            result = {"target": target, "status": "insufficient_reserved_tracks", "tracks": len(tracks)}
        else:
            x, y = [np.array([point[track][0] for track in tracks]) for point in points]
            pixels = np.array([points[1][track][1] for track in tracks])
            arms = {}
            for name, archive in [("registered", data), ("native", raw)]:
                source_e, target_e = np.eye(4), np.eye(4)
                source_e[:3], target_e[:3] = archive["extrinsics"][[i, j]]
                transform = target_e @ np.linalg.inv(source_e)
                scale = float(np.median(data["depth"][i] / raw["depth"][i])) if name == "native" else 1.0
                arms[name] = (1.0, transform[:3, :3], transform[:3, 3] * scale)
            arms["rigid"] = robust_fit(x[~held], y[~held])
            arms["similarity"] = robust_fit(x[~held], y[~held], similarity=True)
            result = {"target": target, "frames": [a, b], "fit_tracks": int((~held).sum()), "reserved_tracks": int(held.sum()), "arms": {}}
            for name, (scale, rotation, translation) in arms.items():
                projected_points = scale * x[held] @ rotation.T + translation
                projected = projected_points @ data["intrinsics"][j].T
                uv = projected[:, :2] / np.maximum(projected[:, 2:], 1e-8)
                pixel_error = np.linalg.norm(uv - pixels[held], axis=1)
                residual = np.linalg.norm(projected_points - y[held], axis=1) / y[held, 2]
                result["arms"][name] = {
                    "scale": scale,
                    "median_reprojection_error_processed_px": float(np.median(pixel_error)),
                    "p95_reprojection_error_processed_px": float(np.quantile(pixel_error, 0.95)),
                    "median_relative_3d_residual": float(np.median(residual)),
                }
        with log.open("a") as stream:
            stream.write(json.dumps(result) + "\n")
            stream.flush()
        print(f"target: {target}\nstatus: saved", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--registered", type=Path, required=True)
    parser.add_argument("--colmap-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.native, args.registered, args.colmap_model, args.output)


if __name__ == "__main__":
    main()
