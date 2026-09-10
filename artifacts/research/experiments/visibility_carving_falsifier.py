"""Test cross-view free-space evidence against the unchanged exported mesh.

Run as a module with --source <registered run> --output <new experiment>.
Uses only training frames, preserves all inputs, checkpoints every prediction
window, and retains separate exports and 100-view reports for each fixed arm.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .local_pose_falsifier import filtered_depth


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_checkpoint(path, **arrays):
    temporary = path.with_suffix(".temporary.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def carve(source, output):
    config = json.loads((source / "inference.json").read_text())
    if not config.get("poses_global"):
        raise ValueError("Use a registered source with shared world coordinates")
    asset = source / "model/property.glb"
    files = sorted((source / "windows").glob("*.npz"))
    signature = {
        "version": 1,
        "source": str(source.resolve()),
        "asset_sha256": sha256(asset),
        "inference_sha256": sha256(source / "inference.json"),
        "cameras_sha256": sha256(source / "model/cameras.json"),
        "windows": {p.name: sha256(p) for p in files},
        "support_tolerance": 0.03,
        "free_space_margin": 0.08,
        "confidence_quantile": 0.75,
        "temporal_bin_frames": 4,
        "held_out_rule": "Exclude all frame IDs ending in 5",
        "interpretation": "Visibility evidence from predicted depth, not ground truth",
    }
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "input.json"
    if marker.exists():
        if json.loads(marker.read_text()) != signature:
            raise ValueError("Experiment inputs changed; choose another --output")
    else:
        with marker.open("x") as stream:
            json.dump(signature, stream, indent=2)
    mesh = trimesh.load(asset, force="mesh", process=False, skip_materials=True)
    centers = np.asarray(mesh.vertices)[mesh.faces].mean(axis=1)
    centers[:, 1:] *= -1  # The evidence cameras use OpenCV axes.
    index = cKDTree(centers)
    cameras = json.loads((source / "model/cameras.json").read_text())
    state_path = output / "votes.npz"
    size = len(centers)
    state = {
        "support": np.zeros(size, np.uint16),
        "contradiction": np.zeros(size, np.uint16),
        "last_support_bin": np.full(size, -1, np.int16),
        "last_contradiction_bin": np.full(size, -1, np.int16),
        "completed_window": np.array(-1),
    }
    if state_path.exists():
        state = dict(np.load(state_path))
    started = time.monotonic()
    for window, file in enumerate(files):
        if window <= int(state["completed_window"]):
            continue
        data = dict(np.load(file))
        training = data["frame_ids"] % 10 != 5
        data = {name: array[training] for name, array in data.items()}
        chosen = [c for c in cameras if c["window"] == window and c["frame"] % 10 != 5]
        for camera in chosen:
            frame = camera["frame"]
            i = int(np.flatnonzero(data["frame_ids"] == frame)[0])
            depth = filtered_depth(data, i)
            confidence = data["confidence"][i]
            depth[confidence < np.quantile(confidence, 0.75)] = 0
            if not np.any(depth > 0):
                continue
            c2w = np.asarray(camera["camera_to_world"])
            e = np.linalg.inv(c2w)
            k = np.asarray(camera["intrinsics"])
            h, w = depth.shape
            corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]])
            radius = float(depth.max() * np.linalg.norm(corners @ np.linalg.inv(k).T, axis=1).max())
            candidates = np.asarray(index.query_ball_point(c2w[:3, 3], radius), dtype=np.int64)
            if not len(candidates):
                continue
            points = centers[candidates] @ e[:3, :3].T + e[:3, 3]
            projected = points @ k.T
            uv = projected[:, :2] / np.maximum(projected[:, 2:], 1e-8)
            visible = ((points[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1)
                       & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1))
            candidates, points, uv = candidates[visible], points[visible], uv[visible]
            # OpenCV remap has a 32767-pixel dimension limit; sample bounded batches.
            observed = np.empty(len(candidates), np.float32)
            for low in range(0, len(candidates), 16000):
                pixels = uv[low:low + 16000].astype(np.float32)
                observed[low:low + len(pixels)] = cv2.remap(
                    depth, pixels[:, 0, None], pixels[:, 1, None], cv2.INTER_NEAREST
                ).ravel()
            good = observed > 0
            supported = good & (np.abs(points[:, 2] - observed) < 0.03 * observed)
            contradicted = good & (points[:, 2] < 0.92 * observed)
            temporal_bin = frame // 4
            for name, selection in [("support", supported), ("contradiction", contradicted)]:
                selected = candidates[selection]
                last = state[f"last_{name}_bin"]
                selected = selected[last[selected] != temporal_bin]
                state[name][selected] += 1
                last[selected] = temporal_bin
        state["completed_window"] = np.array(window)
        save_checkpoint(state_path, **state)
        print(f"windows_processed: {window + 1}\nseconds: {time.monotonic() - started:.1f}", flush=True)

    support, contradiction = state["support"], state["contradiction"]
    masks = {
        "unsupported": (contradiction >= 3) & (support == 0),
        "majority": (contradiction >= 3) & (contradiction > 2 * support),
        "equal_weight": (contradiction >= 3) & (contradiction > support),
    }
    from lingbot_map.reconstruction.materials import unlit_materials
    from lingbot_map.reconstruction.validation import validate

    results = {}
    for name, remove in masks.items():
        arm = output / name
        model = arm / "model"
        model.mkdir(parents=True, exist_ok=True)
        for relative in ["windows", "input.json", "inference.json", "frames"]:
            destination = arm / relative
            if not destination.exists():
                destination.symlink_to((source / relative).resolve())
        for relative in ["cameras.json", "alignment.json"]:
            destination = model / relative
            if not destination.exists():
                destination.symlink_to((source / "model" / relative).resolve())
        target = model / "property.glb"
        if not target.exists():
            carved = trimesh.Trimesh(
                vertices=mesh.vertices.copy(), faces=mesh.faces[~remove].copy(),
                vertex_colors=mesh.visual.vertex_colors.copy(), process=False,
            )
            carved.remove_unreferenced_vertices()
            carved.export(target, tree_postprocessor=unlit_materials)
        report_path = model / "web-render-validation.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else validate(arm, asset_name="property.glb")
        results[name] = {
            "triangles_removed": int(remove.sum()),
            "triangles_retained": int((~remove).sum()),
            "asset_sha256": sha256(target),
            "median_depth_support": report["median_supported_depth_fraction"],
            "minimum_depth_support": report["minimum_supported_depth_fraction"],
            "median_coverage": report["median_rendered_coverage"],
            "median_rgb_error": report["median_absolute_rgb_error_visible"],
            "failed_views": [v["frame"] for v in report["views"] if v["supported_depth_fraction"] < 0.4],
            "gate": report["view_consistency_gate"],
        }
        with (output / "results.json").open("w") as stream:
            json.dump(results, stream, indent=2)
        print(f"arm: {name}\nfailed_views: {len(results[name]['failed_views'])}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    carve(args.source, args.output)


if __name__ == "__main__":
    main()
