"""Export captured RGB and registered cameras for native Brush training."""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from artifacts.research.experiments.mapanything_mac import digest


def nerf_frame(camera, filename, width, height):
    """Convert OpenCV C2W to NeRF camera axes; retain the same world frame."""
    pose = np.array(camera["camera_to_world"], dtype=np.float64, copy=True)
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3, rtol=0):
        raise ValueError("Camera rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1, atol=1e-3, rtol=0):
        raise ValueError("Camera pose includes reflection or scale")
    if not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError("Invalid homogeneous camera pose")
    # Accumulated float32 registration transforms can slightly leave SO(3).
    # Project only near-rigid inputs; never absorb material scale or reflection.
    left, _, right = np.linalg.svd(rotation)
    pose[:3, :3] = left @ right
    intrinsics = np.asarray(camera["intrinsics"])
    return {
        "file_path": filename,
        "transform_matrix": (pose @ np.diag([1, -1, -1, 1])).tolist(),
        "w": width,
        "h": height,
        "fl_x": float(intrinsics[0, 0]),
        "fl_y": float(intrinsics[1, 1]),
        "cx": float(intrinsics[0, 2]),
        "cy": float(intrinsics[1, 2]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=120000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing datasets; use a new output directory")
    if args.points < 100:
        parser.error("Require at least 100 initialization points")
    cameras = json.loads((args.source / "model/cameras.json").read_text())
    manifest = json.loads((args.source / "input.json").read_text())
    if not any(c["held_out_from_fusion"] for c in cameras):
        parser.error("A reserved evaluation split is required")
    files = sorted((args.source / "windows").glob("*.npz"))
    cloud_path = args.source / "model/observed-points.ply"
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    if not len(cloud.points) or not cloud.has_colors():
        parser.error("A colored training-surface point cloud is required")
    count = min(args.points, len(cloud.points))
    indices = np.sort(
        np.random.default_rng(42).choice(len(cloud.points), count, replace=False)
    )
    initial = cloud.select_by_index(indices)
    args.output.mkdir(parents=True)
    (args.output / "images").mkdir()
    initial_path = args.output / "initial.ply"
    if not o3d.io.write_point_cloud(str(initial_path), initial):
        raise RuntimeError("Could not save initialization cloud")
    splits = {"train": [], "val": []}
    assignments, source_windows, output_images = [], {}, {}
    previous, data = None, None
    for camera in cameras:
        path = files[camera["window"]]
        if path != previous:
            with np.load(path) as archive:
                data = dict(archive)
            source_windows[path.name] = digest(path)
            previous = path
        frame = camera["frame"]
        index = int(np.flatnonzero(data["frame_ids"] == frame)[0])
        if not np.allclose(data["intrinsics"][index], camera["intrinsics"], atol=1e-5):
            raise ValueError(
                "Camera calibration does not match the image's owning window"
            )
        original = manifest["frames"][frame].get("original_frame", frame)
        filename = f"images/{original:06d}.png"
        path = args.output / filename
        Image.fromarray(data["rgb"][index]).save(path)
        output_images[filename] = digest(path)
        height, width = data["rgb"][index].shape[:2]
        split = "val" if camera["held_out_from_fusion"] else "train"
        exported = nerf_frame(camera, filename, width, height)
        splits[split].append(exported)
        restored = np.asarray(exported["transform_matrix"]) @ np.diag([1, -1, -1, 1])
        correction = float(np.max(np.abs(restored - camera["camera_to_world"])))
        assignments.append(
            {
                "frame": frame,
                "original_frame": original,
                "split": split,
                "rotation_projection_max_abs": correction,
            }
        )
    for split, frames in splits.items():
        payload = {
            "camera_model": "OPENCV",
            "ply_file_path": "initial.ply",
            "frames": frames,
        }
        with (args.output / f"transforms_{split}.json").open("x") as stream:
            json.dump(payload, stream, indent=2)
    report = {
        "source": str(args.source.resolve()),
        "source_input_sha256": digest(args.source / "input.json"),
        "source_cameras_sha256": digest(args.source / "model/cameras.json"),
        "source_cloud_sha256": digest(cloud_path),
        "source_window_sha256": source_windows,
        "source_points": len(cloud.points),
        "initial_points": count,
        "initial_sha256": digest(initial_path),
        "image_sha256": output_images,
        "train_frames": len(splits["train"]),
        "evaluation_frames": len(splits["val"]),
        "assignments": assignments,
        "camera_convention": "NeRF/OpenGL C2W; Brush loader restores OpenCV camera axes",
        "point_cloud_coordinates": "Unchanged source OpenCV world frame",
        "metric_accuracy_verified": False,
        "interpretation": "Evaluation RGB excluded from Brush training; poses and initialization derive from the existing reconstruction, whose learned inference saw all frames.",
    }
    with (args.output / "provenance.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "source_points",
                    "initial_points",
                    "train_frames",
                    "evaluation_frames",
                    "metric_accuracy_verified",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
