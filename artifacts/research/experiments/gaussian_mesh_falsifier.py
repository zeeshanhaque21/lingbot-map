"""Fuse Gaussian-derived depth with captured RGB and check full/app meshes."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from artifacts.research.experiments.mapanything_mac import digest
from artifacts.research.experiments.photometric_depth_falsifier import publish
from lingbot_map.reconstruction.fusion import fuse
from lingbot_map.reconstruction.validation import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train-depth", type=Path, required=True)
    parser.add_argument("--eval-depth", type=Path, required=True)
    parser.add_argument("--manifest-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=["center_depth", "median_center_depth", "median_depth"],
        required=True,
    )
    args = parser.parse_args()
    provenance = json.loads((args.dataset / "provenance.json").read_text())
    manifest = json.loads((args.manifest_source / "input.json").read_text())
    frames, records, signatures = {}, [], {}
    for split, directory in [("train", args.train_depth), ("val", args.eval_depth)]:
        signatures[split] = json.loads((directory / "signature.json").read_text())
        if signatures[split]["split"] != split:
            raise ValueError("Depth directory belongs to the wrong split")
        trace = {
            entry["frame"]: entry
            for entry in (
                json.loads(line)
                for line in (directory / "trace.jsonl").read_text().splitlines()
            )
        }
        for frame in json.loads(
            (args.dataset / f"transforms_{split}.json").read_text()
        )["frames"]:
            number = int(Path(frame["file_path"]).stem)
            if number in frames:
                raise ValueError("Training and evaluation frames overlap")
            path = directory / f"{number:06d}.npz"
            if number not in trace or digest(path) != trace[number]["sha256"]:
                raise ValueError("Depth frame is incomplete or changed")
            image_path = args.dataset / frame["file_path"]
            if digest(image_path) != provenance["image_sha256"][frame["file_path"]]:
                raise ValueError("Captured image changed after dataset preparation")
            data = dict(np.load(path))
            raw_depth = data[args.method]
            valid = (data["alpha"] > 0.9) & (raw_depth > 0) & np.isfinite(raw_depth)
            pose = np.asarray(frame["transform_matrix"]) @ np.diag([1, -1, -1, 1])
            frames[number] = {
                "rgb": np.asarray(Image.open(image_path).convert("RGB")),
                "depth": np.where(valid, raw_depth, 0).astype(np.float32),
                "confidence": np.where(valid, data["alpha"], 0).astype(np.float32),
                "intrinsics": np.array(
                    [
                        [frame["fl_x"], 0, frame["cx"]],
                        [0, frame["fl_y"], frame["cy"]],
                        [0, 0, 1],
                    ],
                    np.float32,
                ),
                "extrinsics": np.linalg.inv(pose)[:3].astype(np.float32),
                "split": split,
            }
            records.append(
                {
                    "frame": number,
                    "split": split,
                    "depth_sha256": trace[number]["sha256"],
                    "alpha_supported_fraction": float(valid.mean()),
                }
            )
    if signatures["train"]["ply_sha256"] != signatures["val"]["ply_sha256"]:
        raise ValueError("Training and evaluation depth use different Gaussian models")
    ordered = []
    for index, frame in enumerate(manifest["frames"]):
        number = frame.get("original_frame", index)
        if number not in frames or (frames[number]["split"] == "val") != (
            index % 10 == 5
        ):
            raise ValueError("Manifest does not preserve the reserved frame assignment")
        ordered.append(frames[number])
    if len(ordered) != len(frames):
        raise ValueError("Dataset and manifest contain different frames")
    data = {
        name: np.stack([frame[name] for frame in ordered])
        for name in ["rgb", "depth", "confidence", "intrinsics", "extrinsics"]
    }
    data.update(
        frame_ids=np.arange(len(ordered)),
        registered=np.ones(len(ordered), bool),
        pixel_center_offset=np.float32(0.5),
    )
    inference = {
        "window": len(ordered),
        "overlap": min(8, len(ordered) - 1),
        "poses_global": True,
        "pose_source": "COLMAP with Gaussian-derived depth",
        "method": args.method,
        "pixel_center_offset": 0.5,
        "alpha_minimum": 0.9,
        "gaussian_depth_signatures": signatures,
        "implementation_sha256": {
            name: digest(Path("lingbot_map/reconstruction") / name)
            for name in ["fusion.py", "geometry.py", "validation.py", "export.py"]
        },
        "dataset_provenance_sha256": digest(args.dataset / "provenance.json"),
        "metric_accuracy_verified": False,
    }
    publish(args.output, data, manifest, inference)
    fuse(args.output)
    report = {
        "method": args.method,
        "source_frames": records,
        "full": validate(args.output),
        "app": validate(args.output, asset_name="property.glb"),
        "interpretation": "TSDF uses captured training RGB and Gaussian-derived depths. Reserved RGB does not enter fusion or splat training, but COLMAP used all images. Validation depth is model-derived and does not establish independently measured accuracy.",
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "method": args.method,
                "full_gate": report["full"]["view_consistency_gate"],
                "app_gate": report["app"]["view_consistency_gate"],
                "app_coverage": report["app"]["median_rendered_coverage"],
                "app_depth_support": report["app"]["median_supported_depth_fraction"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
