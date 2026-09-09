"""Audit Gaussian-renderer agreement and depth against existing sparse geometry."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from artifacts.research.experiments.mapanything_mac import digest
from lingbot_map.reconstruction.colmap_io import read_model
from lingbot_map.reconstruction.geometry import preprocessing_transform


def sample(array, pixels):
    # Brush's raster pixel centers are (x+0.5, y+0.5).
    return cv2.remap(
        array.astype(np.float32),
        (pixels[:, 0] - 0.5).astype(np.float32)[:, None],
        (pixels[:, 1] - 0.5).astype(np.float32)[:, None],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )[:, 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--colmap-text", type=Path, required=True)
    parser.add_argument("--native-renders", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing audits; use a new output directory")
    args.output.mkdir(parents=True)
    images, points = read_model(args.colmap_text)
    transforms = json.loads((args.dataset / "transforms_val.json").read_text())
    rows, figures = [], []
    for frame in transforms["frames"]:
        name = Path(frame["file_path"]).stem
        path = args.depth / f"{name}.npz"
        data = dict(np.load(path))
        native = (
            np.asarray(Image.open(args.native_renders / f"{name}.png").convert("RGB"))
            / 255.0
        )
        difference = np.abs(data["rgb"].clip(0, 1) - native)
        visible = (data["alpha"] > 0.9) & (data["median_depth"] > 0)
        narrow = (
            visible
            & (data["center_iqr"] < 0.05 * data["median_depth"])
            & (data["median_sigma"] < 0.05 * data["median_depth"])
        )
        camera = images[f"{name}.jpg"]
        model = camera["camera"]
        transform = preprocessing_transform(
            model["width"], model["height"], frame["w"], frame["h"]
        )
        pixels, depths = [], []
        for x, y, identifier in camera["observations"]:
            point = points.get(int(identifier))
            if point is None or point["track_length"] < 3:
                continue
            position = (
                camera["extrinsics"][:3, :3] @ point["xyz"]
                + camera["extrinsics"][:3, 3]
            )
            if position[2] <= 0:
                continue
            projected = model["intrinsics"] @ position
            if np.linalg.norm(projected[:2] / projected[2] - [x, y]) > 2:
                continue
            pixel = (transform @ [x, y, 1])[:2]
            if not (
                0.5 <= pixel[0] < frame["w"] - 0.5
                and 0.5 <= pixel[1] < frame["h"] - 0.5
            ):
                continue
            pixels.append(pixel)
            depths.append(position[2])
        pixels, depths = np.asarray(pixels), np.asarray(depths)
        if not len(depths):
            raise ValueError(f"No usable triangulated observations for {name}")
        covered = sample(data["alpha"], pixels) > 0.9
        estimates = {}
        for method in ["center_depth", "depth", "median_center_depth", "median_depth"]:
            prediction = sample(data[method], pixels)
            valid = covered & (prediction > 0)
            error = np.abs(prediction - depths) / depths
            estimates[method] = {
                "coverage_fraction": float(valid.mean()),
                "within_5percent_all_tracks": float(np.mean(valid & (error < 0.05))),
                "median_relative_error_covered": float(np.median(error[valid]))
                if valid.any()
                else None,
                "p95_relative_error_covered": float(np.quantile(error[valid], 0.95))
                if valid.any()
                else None,
            }
        narrow_tracks = sample(narrow.astype(np.float32), pixels) > 0.999
        error = np.abs(sample(data["median_depth"], pixels) - depths) / depths
        estimates["median_depth_narrow"] = {
            "coverage_fraction": float(narrow_tracks.mean()),
            "within_5percent_all_tracks": float(
                np.mean(narrow_tracks & (error < 0.05))
            ),
            "median_relative_error_covered": float(np.median(error[narrow_tracks]))
            if narrow_tracks.any()
            else None,
        }
        row = {
            "frame": int(name),
            "depth_sha256": digest(path),
            "native_png_sha256": digest(args.native_renders / f"{name}.png"),
            "renderer_rgb_mae": float(difference.mean()),
            "renderer_rgb_p99": float(np.quantile(difference, 0.99)),
            "renderer_rgb_max": float(difference.max()),
            "sparse_reference_points": len(depths),
            "alpha_over_09": float(visible.mean()),
            "narrow_depth_fraction": float(narrow.mean()),
            "median_relative_center_iqr": float(
                np.median(data["center_iqr"][visible] / data["median_depth"][visible])
            ),
            "estimates": estimates,
        }
        rows.append(row)
        reference = np.asarray(
            Image.open(args.dataset / frame["file_path"]).convert("RGB")
        )
        h, w = reference.shape[:2]
        depth = data["median_depth"]
        lo, hi = np.quantile(depth[visible], [0.02, 0.98])
        normalized = (
            (np.log(np.maximum(depth, lo)) - np.log(lo)) / (np.log(hi) - np.log(lo))
        ).clip(0, 1)
        colored = cv2.applyColorMap(
            (normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
        )[..., ::-1]
        colored[~visible] = 0
        width_map = (data["center_iqr"] / np.maximum(depth, 1e-8) / 0.25).clip(0, 1)
        dispersion = cv2.applyColorMap(
            (width_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
        )[..., ::-1]
        dispersion[~visible] = 0
        figure = Image.new("RGB", (3 * w, h + 28), (24, 24, 24))
        for column, (title, array) in enumerate(
            [
                (f"Captured {name}", reference),
                ("Median depth; per-view log scale", colored),
                ("Depth IQR: black 0%, bright >=25%", dispersion),
            ]
        ):
            ImageDraw.Draw(figure).text((column * w + 5, 7), title, fill="white")
            figure.paste(Image.fromarray(array), (column * w, 28))
        figures.append(figure)
    sheet = Image.new(
        "RGB",
        (figures[0].width, sum(figure.height for figure in figures)),
        (24, 24, 24),
    )
    y = 0
    for figure in figures:
        sheet.paste(figure, (0, y))
        y += figure.height
    sheet.save(args.output / "depth-audit.jpg", quality=94)
    report = {
        "rows": rows,
        "depth_signature": json.loads((args.depth / "signature.json").read_text()),
        "colmap_sha256": {
            name: digest(args.colmap_text / name)
            for name in ["images.txt", "cameras.txt", "points3D.txt"]
        },
        "metric_accuracy_verified": False,
        "ready_for_verified_property_listing": False,
        "interpretation": "Renderer RGB agreement checks implementation compatibility. Sparse COLMAP points participated in camera estimation and seed geometry; they are not independent surveyed truth. Median depth uses center-depth ordering and each selected Gaussian's conditional projected Z. IQR and component width are diagnostics, not calibrated confidence probabilities. Center-depth expectation is the unchanged Gaussian extraction baseline.",
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
