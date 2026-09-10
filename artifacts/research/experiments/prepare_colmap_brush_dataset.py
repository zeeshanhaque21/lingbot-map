"""Build a Brush comparison with COLMAP cameras and training-track seed points."""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from artifacts.research.experiments.mapanything_mac import digest
from artifacts.research.experiments.prepare_brush_dataset import nerf_frame
from lingbot_map.reconstruction.colmap_io import read_model
from lingbot_map.reconstruction.geometry import preprocessing_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--colmap-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing datasets; use a new output directory")
    original = json.loads((args.dataset / "provenance.json").read_text())
    images, points = read_model(args.colmap_text)
    splits, point_colors, observations = {}, defaultdict(list), []
    for split in ["train", "val"]:
        prior = json.loads((args.dataset / f"transforms_{split}.json").read_text())
        frames = []
        for old in prior["frames"]:
            image_path = args.dataset / old["file_path"]
            key = image_path.with_suffix(".jpg").name
            if key not in images:
                raise ValueError(f"COLMAP camera missing for {key}")
            item = images[key]
            model = item["camera"]
            width, height = old["w"], old["h"]
            transform = preprocessing_transform(
                model["width"], model["height"], width, height
            )
            intrinsic = transform @ model["intrinsics"]
            extrinsic = np.eye(4)
            extrinsic[:3] = item["extrinsics"]
            camera = {
                "camera_to_world": np.linalg.inv(extrinsic),
                "intrinsics": intrinsic,
            }
            frames.append(nerf_frame(camera, old["file_path"], width, height))
            rgb = np.asarray(Image.open(image_path).convert("RGB")) / 255.0
            accepted_errors = []
            seen = set()
            for x, y, point_id in item["observations"]:
                point_id = int(point_id)
                if point_id not in points or point_id in seen:
                    continue
                point = points[point_id]["xyz"]
                camera_point = extrinsic[:3, :3] @ point + extrinsic[:3, 3]
                if camera_point[2] <= 0:
                    continue
                projected = model["intrinsics"] @ camera_point
                error = float(np.linalg.norm(projected[:2] / projected[2] - [x, y]))
                pixel = transform @ [x, y, 1]
                u, v = np.rint(pixel[:2]).astype(int)
                if error > 2 or not (0 <= u < width and 0 <= v < height):
                    continue
                accepted_errors.append(error)
                seen.add(point_id)
                if split == "train":
                    point_colors[point_id].append(rgb[v, u])
            observations.append(
                {
                    "frame": int(image_path.stem),
                    "split": split,
                    "accepted_tracks": len(accepted_errors),
                    "median_reprojection_error_original_pixels": float(
                        np.median(accepted_errors)
                    )
                    if accepted_errors
                    else None,
                }
            )
        splits[split] = {
            "camera_model": "OPENCV",
            "ply_file_path": "initial.ply",
            "frames": frames,
        }
    selected = sorted(key for key, values in point_colors.items() if len(values) >= 2)
    if len(selected) < 100:
        raise ValueError("Insufficient points observed in at least two training images")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector([points[key]["xyz"] for key in selected])
    cloud.colors = o3d.utility.Vector3dVector(
        [np.mean(point_colors[key], axis=0) for key in selected]
    )
    args.output.mkdir(parents=True)
    shutil.copytree(args.dataset / "images", args.output / "images")
    for filename, expected in original["image_sha256"].items():
        if digest(args.output / filename) != expected:
            raise ValueError(f"Image changed: {filename}")
    initial_path = args.output / "initial.ply"
    if not o3d.io.write_point_cloud(str(initial_path), cloud):
        raise RuntimeError("Could not save initialization cloud")
    for split, data in splits.items():
        (args.output / f"transforms_{split}.json").write_text(
            json.dumps(data, indent=2)
        )
    report = {
        "parent_dataset": str(args.dataset.resolve()),
        "parent_provenance_sha256": digest(args.dataset / "provenance.json"),
        "colmap_text": str(args.colmap_text.resolve()),
        "colmap_sha256": {
            name: digest(args.colmap_text / name)
            for name in ["images.txt", "cameras.txt", "points3D.txt"]
        },
        "initial_points": len(selected),
        "initial_sha256": digest(initial_path),
        "image_sha256": original["image_sha256"],
        "train_frames": len(splits["train"]["frames"]),
        "evaluation_frames": len(splits["val"]["frames"]),
        "assignments": original["assignments"],
        "observations": observations,
        "camera_convention": "NeRF/OpenGL C2W; unchanged COLMAP world coordinates",
        "metric_accuracy_verified": False,
        "interpretation": "Camera and seed initialization change together. Seed colors use only training RGB and points require two training observations within 2 original pixels. COLMAP itself used all images; reserved RGB is excluded only from Brush optimization.",
    }
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                key: report[key]
                for key in ["initial_points", "train_frames", "evaluation_frames"]
            }
        )
    )


if __name__ == "__main__":
    main()
