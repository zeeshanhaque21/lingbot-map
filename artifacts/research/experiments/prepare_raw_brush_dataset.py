"""Retain COLMAP seed geometry while using the original extracted RGB resolution."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from artifacts.research.experiments.mapanything_mac import digest
from artifacts.research.experiments.prepare_brush_dataset import nerf_frame
from lingbot_map.reconstruction.colmap_io import read_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest-source", type=Path, required=True)
    parser.add_argument("--colmap-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing datasets; use a new output directory")
    parent = json.loads((args.dataset / "provenance.json").read_text())
    manifest = json.loads((args.manifest_source / "input.json").read_text())
    source_frames = {
        frame.get("original_frame", index): frame
        for index, frame in enumerate(manifest["frames"])
    }
    images, _ = read_model(args.colmap_text)
    args.output.mkdir(parents=True)
    (args.output / "images").mkdir()
    if digest(args.dataset / "initial.ply") != parent["initial_sha256"]:
        raise ValueError("Parent seed geometry changed")
    shutil.copy2(args.dataset / "initial.ply", args.output / "initial.ply")
    hashes, source_hashes, assignments, counts = {}, {}, [], {}
    for split in ["train", "val"]:
        frames = []
        for previous in json.loads(
            (args.dataset / f"transforms_{split}.json").read_text()
        )["frames"]:
            number = int(Path(previous["file_path"]).stem)
            source = Path(source_frames[number]["file"])
            if not source.is_absolute():
                source = args.manifest_source / source
            image = Image.open(source).convert("RGB")
            camera = images[f"{number:06d}.jpg"]
            model = camera["camera"]
            if image.size != (model["width"], model["height"]):
                raise ValueError("Raw image dimensions differ from COLMAP calibration")
            world_to_camera = np.eye(4)
            world_to_camera[:3] = camera["extrinsics"]
            pose = np.linalg.inv(world_to_camera)
            old_pose = np.asarray(previous["transform_matrix"]) @ np.diag(
                [1, -1, -1, 1]
            )
            if not np.allclose(pose, old_pose, atol=1e-6, rtol=0):
                raise ValueError(
                    "Raw and processed trials must retain identical cameras"
                )
            filename = f"images/{number:06d}.png"
            image.save(args.output / filename)
            frames.append(
                nerf_frame(
                    {"camera_to_world": pose, "intrinsics": model["intrinsics"]},
                    filename,
                    *image.size,
                )
            )
            hashes[filename] = digest(args.output / filename)
            source_hashes[str(source)] = digest(source)
            assignments.append(
                {"frame": number, "original_frame": number, "split": split}
            )
        counts[split] = len(frames)
        (args.output / f"transforms_{split}.json").write_text(
            json.dumps(
                {
                    "camera_model": "OPENCV",
                    "ply_file_path": "initial.ply",
                    "frames": frames,
                },
                indent=2,
            )
        )
    report = {
        "parent_dataset": str(args.dataset.resolve()),
        "parent_provenance_sha256": digest(args.dataset / "provenance.json"),
        "initial_points": parent["initial_points"],
        "initial_sha256": digest(args.output / "initial.ply"),
        "image_sha256": hashes,
        "source_image_sha256": source_hashes,
        "colmap_sha256": {
            name: digest(args.colmap_text / name)
            for name in ["cameras.txt", "images.txt", "points3D.txt"]
        },
        "train_frames": counts["train"],
        "evaluation_frames": counts["val"],
        "assignments": assignments,
        "metric_accuracy_verified": False,
        "interpretation": "Uses original extracted JPEG pixels at COLMAP's calibrated resolution, decoded losslessly to PNG. Camera poses, seed XYZ/colors, and training/evaluation assignment remain identical to the processed-resolution COLMAP trial. This is not the original 4K video resolution.",
    }
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "initial_points": parent["initial_points"],
                "train": counts["train"],
                "val": counts["val"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
