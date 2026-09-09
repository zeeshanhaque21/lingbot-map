"""Preserve full-capture RGB and calibrated cameras in the shared experiment layout."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from artifacts.research.experiments.gaussian_depth_renderer import digest
from lingbot_map.reconstruction.colmap_io import read_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--colmap-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing datasets; use a new output directory")
    manifest = json.loads((args.capture / "input.json").read_text())
    images, _ = read_model(args.colmap_text)
    missing = [
        frame["file"]
        for frame in manifest["frames"]
        if Path(frame["file"]).name not in images
    ]
    if missing:
        raise ValueError(f"Capture cameras are missing from COLMAP: {missing[:10]}")
    identifiers = [frame["id"] for frame in manifest["frames"]]
    if len(set(identifiers)) != len(identifiers) or len(identifiers) < 10:
        raise ValueError("Expected unique capture frame IDs and at least ten images")
    (args.output / "images").mkdir(parents=True)
    splits = {"train": [], "val": []}
    source_hashes, image_hashes = {}, {}
    with (args.output / "images.jsonl").open("x") as trace:
        for frame in manifest["frames"]:
            source = args.capture / frame["file"]
            name = source.stem
            if name != f"{frame['id']:06d}" or source.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError("Expected numbered captured PNG or JPEG images")
            calibration = images[source.name]
            camera = calibration["camera"]
            rgb = Image.open(source).convert("RGB")
            if rgb.size != (camera["width"], camera["height"]):
                raise ValueError(
                    f"Captured pixels and calibration differ for {source.name}"
                )
            relative = f"images/{name}.png"
            destination = args.output / relative
            rgb.save(destination)
            source_hashes[frame["file"]] = digest(source)
            image_hashes[relative] = digest(destination)
            extrinsic = np.vstack([calibration["extrinsics"], [0, 0, 0, 1]])
            pose = np.linalg.inv(extrinsic) @ np.diag([1, -1, -1, 1])
            k = camera["intrinsics"]
            split = "val" if frame["id"] % 10 == 5 else "train"
            splits[split].append(
                {
                    "file_path": relative,
                    "transform_matrix": pose.tolist(),
                    "w": camera["width"],
                    "h": camera["height"],
                    "fl_x": float(k[0, 0]),
                    "fl_y": float(k[1, 1]),
                    "cx": float(k[0, 2]),
                    "cy": float(k[1, 2]),
                    "original_frame": frame["id"],
                    "timestamp_seconds": frame["timestamp_seconds"],
                }
            )
            trace.write(
                json.dumps(
                    {
                        "frame": frame["id"],
                        "split": split,
                        "source_sha256": source_hashes[frame["file"]],
                        "png_sha256": image_hashes[relative],
                    }
                )
                + "\n"
            )
            trace.flush()
            if len(image_hashes) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "images_completed": len(image_hashes),
                            "images_expected": len(identifiers),
                        }
                    ),
                    flush=True,
                )
    for split, frames in splits.items():
        (args.output / f"transforms_{split}.json").write_text(
            json.dumps({"camera_model": "PINHOLE", "frames": frames}, indent=2)
        )
    report = {
        "capture": str(args.capture.resolve()),
        "source_manifest_sha256": digest(args.capture / "input.json"),
        "source_video_sha256": manifest["configuration"].get("source_sha256"),
        "source_kind": manifest["configuration"].get("source_kind", "video"),
        "source_image_sha256": source_hashes,
        "image_sha256": image_hashes,
        "colmap_sha256": {
            name: digest(args.colmap_text / name)
            for name in ("cameras.txt", "images.txt", "points3D.txt")
        },
        "train_frames": len(splits["train"]),
        "evaluation_frames": len(splits["val"]),
        "metric_accuracy_verified": False,
        "interpretation": "Full capture with captured pixels decoded losslessly to PNG and unchanged COLMAP cameras. Every frame ending in 5 is reserved from dense reconstruction and texturing. COLMAP used all images, so reserved-view comparisons are development checks, not independent ground truth. No scene scale or connectivity claim follows from low reprojection error.",
    }
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "training_images": report["train_frames"],
                "reserved_images": report["evaluation_frames"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
