"""Export captured training images and their calibrated sparse tracks for OpenMVS."""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from artifacts.research.experiments.gaussian_depth_renderer import digest
from lingbot_map.reconstruction.colmap_io import captured_image, read_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--colmap-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing datasets; use a new output directory")
    provenance = json.loads((args.dataset / "provenance.json").read_text())
    for name, expected in provenance["colmap_sha256"].items():
        if digest(args.colmap_text / name) != expected:
            raise ValueError("Source calibration changed")
    transforms = json.loads((args.dataset / "transforms_train.json").read_text())
    reserved = {
        Path(frame["file_path"]).stem
        for frame in json.loads((args.dataset / "transforms_val.json").read_text())[
            "frames"
        ]
    }
    source_images, points = read_model(args.colmap_text)
    selected, tracks, colors = [], defaultdict(list), defaultdict(list)
    for frame in transforms["frames"]:
        name = Path(frame["file_path"]).stem
        if name in reserved:
            raise ValueError("Training and reserved images overlap")
        path = args.dataset / frame["file_path"]
        if digest(path) != provenance["image_sha256"][frame["file_path"]]:
            raise ValueError("Captured image changed")
        image = captured_image(source_images, int(name))
        camera = image["camera"]
        expected_pose = np.linalg.inv(
            np.asarray(frame["transform_matrix"]) @ np.diag([1, -1, -1, 1])
        )[:3]
        if not np.allclose(image["extrinsics"], expected_pose, atol=1e-6, rtol=0):
            raise ValueError("COLMAP and dataset camera poses differ")
        expected_k = np.array(
            [
                [frame["fl_x"], 0, frame["cx"]],
                [0, frame["fl_y"], frame["cy"]],
                [0, 0, 1],
            ]
        )
        if not np.allclose(camera["intrinsics"], expected_k, atol=1e-6, rtol=0):
            raise ValueError("Dataset must retain the original COLMAP calibration")
        rgb = np.asarray(Image.open(path).convert("RGB"))
        if rgb.shape[:2] != (camera["height"], camera["width"]):
            raise ValueError("Image dimensions differ from calibration")
        observations = []
        image_id = len(selected) + 1
        for x, y, raw_identifier in image["observations"]:
            identifier = int(raw_identifier)
            point = points.get(identifier)
            if point is None or point["error"] > 2:
                continue
            position = (
                image["extrinsics"][:, :3] @ point["xyz"] + image["extrinsics"][:, 3]
            )
            if position[2] <= 0 or not (
                0 <= x < camera["width"] and 0 <= y < camera["height"]
            ):
                continue
            tracks[identifier].append((image_id, len(observations)))
            colors[identifier].append(
                rgb[min(round(y), rgb.shape[0] - 1), min(round(x), rgb.shape[1] - 1)]
            )
            observations.append((x, y, identifier))
        selected.append((name, path, image, observations))
    retained = {identifier for identifier, track in tracks.items() if len(track) >= 2}
    if not retained or len(selected) < 3:
        raise ValueError("Insufficient training-only sparse geometry")
    sparse = args.output / "sparse"
    sparse.mkdir(parents=True)
    (args.output / "images").mkdir()
    camera_lines, image_lines, image_hashes = [], [], {}
    for image_id, (name, path, image, observations) in enumerate(selected, 1):
        camera, extrinsics = image["camera"], image["extrinsics"]
        k = camera["intrinsics"]
        # One explicit camera per image avoids silently tying different calibrations.
        camera_lines.append(
            f"{image_id} PINHOLE {camera['width']} {camera['height']} {k[0, 0]:.17g} {k[1, 1]:.17g} {k[0, 2]:.17g} {k[1, 2]:.17g}"
        )
        quaternion = Rotation.from_matrix(extrinsics[:, :3]).as_quat()[[3, 0, 1, 2]]
        pose = " ".join(f"{x:.17g}" for x in np.r_[quaternion, extrinsics[:, 3]])
        image_lines.append(f"{image_id} {pose} {image_id} {name}.png")
        image_lines.append(
            " ".join(
                f"{x:.17g} {y:.17g} {identifier if identifier in retained else -1}"
                for x, y, identifier in observations
            )
        )
        destination = args.output / "images" / f"{name}.png"
        shutil.copy2(path, destination)
        image_hashes[destination.name] = digest(destination)
    point_lines = []
    for identifier in sorted(retained):
        point = points[identifier]
        xyz = " ".join(f"{value:.17g}" for value in point["xyz"])
        rgb = " ".join(
            map(str, np.median(colors[identifier], axis=0).round().astype(int))
        )
        track = " ".join(
            f"{image_id} {index}" for image_id, index in tracks[identifier]
        )
        point_lines.append(f"{identifier} {xyz} {rgb} {point['error']:.17g} {track}")
    for name, lines in [
        ("cameras.txt", camera_lines),
        ("images.txt", image_lines),
        ("points3D.txt", point_lines),
    ]:
        (sparse / name).write_text("\n".join(lines) + "\n")
    recovered_images, recovered_points = read_model(sparse)
    if len(recovered_images) != len(selected) or set(recovered_points) != retained:
        raise ValueError("Exported sparse model failed round-trip verification")
    for name, _, original, _ in selected:
        recovered = recovered_images[f"{name}.png"]
        if not np.allclose(
            recovered["extrinsics"], original["extrinsics"], atol=1e-12, rtol=0
        ):
            raise ValueError("Exported camera changed")
    for identifier in retained:
        for image_id, index in tracks[identifier]:
            name = selected[image_id - 1][0]
            if recovered_images[f"{name}.png"]["observations"][index, 2] != identifier:
                raise ValueError("Exported track points to the wrong observation")
    report = {
        "training_images": len(selected),
        "reserved_images_excluded": sorted(reserved),
        "sparse_points": len(retained),
        "image_sha256": image_hashes,
        "source_provenance_sha256": digest(args.dataset / "provenance.json"),
        "source_colmap_sha256": provenance["colmap_sha256"],
        "exported_colmap_sha256": {
            name: digest(sparse / name)
            for name in ("cameras.txt", "images.txt", "points3D.txt")
        },
        "metric_accuracy_verified": False,
        "interpretation": "Dense reconstruction receives only training images and tracks observed in at least two training images. Camera estimation previously used reserved images; this is not an independent surveyed benchmark. Coordinates and calibration remain unchanged.",
    }
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "training_images",
                    "reserved_images_excluded",
                    "sparse_points",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
