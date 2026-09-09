"""Compare a complete generated sweep with raw renders and captured anchors."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from lingbot_map.reconstruction.io import digest, write_json
from lingbot_map.reconstruction.rendering import SurfaceRenderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--refinement", type=Path, required=True)
    parser.add_argument("--stitch", type=Path, required=True)
    parser.add_argument("--baseline-stitch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous comparisons; choose a new output directory")
    config = json.loads((args.sweep / "sweep.json").read_text())
    completion = json.loads((args.refinement / "completion.json").read_text())
    if not completion["success"] or not completion.get(
        "complete_sweep", completion.get("full_spherical_sweep", False)
    ):
        parser.error("Comparison requires a completed inference run of the whole sweep")
    paths = sorted((args.refinement / "frames").glob("*.png"))
    if len(paths) != len(config["frames"]):
        parser.error("Generated frame count differs from the rendered sweep")
    mesh = Path(config["tour"]) / "render-mesh.ply"
    if digest(mesh) != config["mesh_sha256"]:
        parser.error("Rendering mesh changed after sweep export")
    renderer = SurfaceRenderer.from_file(mesh)
    intrinsics = np.asarray(config["intrinsics"])
    intrinsics[:2, 2] += 0.5
    width, height = config["width"], config["height"]
    args.output.mkdir(parents=True)
    rows, anchors = [], []
    for index, (path, frame) in enumerate(zip(paths, config["frames"], strict=True)):
        generated = np.asarray(Image.open(path).convert("RGB"))
        if generated.shape[:2] != (height, width):
            raise ValueError("Generated images changed the calibrated dimensions")
        raw, _, visible = renderer.render(
            intrinsics, np.linalg.inv(frame["camera_to_world"]), width, height
        )
        reference = np.asarray(
            Image.open(args.sweep / "frames" / f"{index:03d}.png").convert("RGB")
        )
        delta = (generated.astype(np.float32) - raw) / 255
        record = {
            "frame": index,
            "observed_fraction": float(visible.mean()),
            "generated_vs_raw_rgb_error_on_observed_pixels": float(
                np.abs(delta[visible]).mean()
            )
            if visible.any()
            else None,
        }
        if frame["captured_anchor"]:
            errors = {}
            for label, image in [("raw", raw), ("generated", generated)]:
                difference = (image.astype(np.float32) - reference) / 255
                mse = float(np.mean(difference**2))
                errors[label] = {
                    "absolute_rgb_error_all_pixels": float(np.abs(difference).mean()),
                    "psnr_all_pixels": float(-10 * np.log10(max(mse, 1e-12))),
                    "absolute_rgb_error_observed_pixels": float(
                        np.abs(difference[visible]).mean()
                    ),
                }
            anchors.append({"frame": index, "comparison": errors})
        rows.append(record)
        sample_indices = (
            (0, 8, 15, 23, 30, 38, 45, 53, 60)
            if config.get("sweep_mode") == "horizontal"
            else (0, 4, 12, 24, 36, 48, 56, 60)
        )
        if index in sample_indices:
            sheet = Image.new("RGB", (width * 3, height + 32), (25, 30, 34))
            draw = ImageDraw.Draw(sheet)
            for column, (label, pixels) in enumerate(
                [
                    (f"Input frame {index}", reference),
                    ("Observed mesh render", raw),
                    ("FixAnything candidate", generated),
                ]
            ):
                draw.text((column * width + 12, 10), label, fill="white")
                sheet.paste(Image.fromarray(pixels), (column * width, 32))
            sheet.save(args.output / f"comparison-{index:03d}.jpg", quality=94)
        write_json(args.output / "frames.json", rows)
    generated_stitch = json.loads((args.stitch / "stitch.json").read_text())
    raw_stitch = json.loads((args.baseline_stitch / "stitch.json").read_text())
    raw_panorama = Image.open(args.baseline_stitch / "panorama.png").convert("RGB")
    refined_panorama = Image.open(args.stitch / "panorama.png").convert("RGB")
    if raw_panorama.size != refined_panorama.size:
        raise ValueError("Panorama comparison requires identical dimensions")
    sheet = Image.new(
        "RGB", (raw_panorama.width, raw_panorama.height * 2 + 64), (25, 30, 34)
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "Unchanged input sweep stitch", fill="white")
    sheet.paste(raw_panorama, (0, 32))
    draw.text(
        (12, raw_panorama.height + 42),
        "FixAnything candidate - generated details need review",
        fill="white",
    )
    sheet.paste(refined_panorama, (0, raw_panorama.height + 64))
    sheet.save(args.output / "panorama-comparison.jpg", quality=94)
    report = {
        "raw_stitch": raw_stitch,
        "generated_stitch": generated_stitch,
        "captured_anchors": anchors,
        "unique_captured_anchor_views": 1,
        "independent_reference_views": 0,
        "mesh_unchanged": True,
        "mesh_sha256": config["mesh_sha256"],
        "generated_frames": len(rows),
        "sweep_mode": config.get("sweep_mode", "spherical"),
        "coverage_target": config.get("coverage_target", "sphere"),
        "interpretation": "Anchor comparison uses the photograph supplied to the model, not held-out ground truth. Changes from raw rendering measure edits, not architectural accuracy. Inspect doors, windows, walls and overlap ghosts before accepting a generated tour.",
        "architectural_fidelity_verified": False,
    }
    write_json(args.output / "comparison.json", report)
    print(
        json.dumps(
            {
                "generated_frames": len(rows),
                "anchors": anchors,
                "raw_overlap_rgb_std": raw_stitch["mean_overlap_rgb_std"],
                "generated_overlap_rgb_std": generated_stitch["mean_overlap_rgb_std"],
                "raw_seam": raw_stitch["wrap_seam_rgb_difference"],
                "generated_seam": generated_stitch["wrap_seam_rgb_difference"],
            }
        )
    )


if __name__ == "__main__":
    main()
