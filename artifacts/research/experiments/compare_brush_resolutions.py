"""Compare two Gaussian models against identical raw-resolution reserved images."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from artifacts.research.experiments.evaluate_brush_trial import image_metrics
from artifacts.research.experiments.mapanything_mac import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing comparisons; use a new output directory")
    frames = json.loads((args.dataset / "transforms_val.json").read_text())["frames"]
    provenance = json.loads((args.dataset / "provenance.json").read_text())
    signatures, traces = {}, {}
    for key, directory in [("low", args.low), ("high", args.high)]:
        signatures[key] = json.loads((directory / "signature.json").read_text())
        traces[key] = {
            row["frame"]: row
            for row in (
                json.loads(line)
                for line in (directory / "trace.jsonl").read_text().splitlines()
            )
        }
        if signatures[key]["transforms_sha256"] != digest(
            args.dataset / "transforms_val.json"
        ):
            raise ValueError("Models were rendered with different cameras")
    if signatures["low"]["algorithm_sha256"] != signatures["high"]["algorithm_sha256"]:
        raise ValueError("Both models must use the same renderer implementation")
    args.output.mkdir(parents=True)
    rows, figures = [], []
    font = ImageFont.load_default(size=14)
    for frame in frames:
        number = int(Path(frame["file_path"]).stem)
        image_path = args.dataset / frame["file_path"]
        if digest(image_path) != provenance["image_sha256"][frame["file_path"]]:
            raise ValueError("Reserved source image changed")
        reference = np.asarray(Image.open(image_path).convert("RGB"))
        row, rendered = {"frame": number, "reference_sha256": digest(image_path)}, {}
        for key, directory in [("low", args.low), ("high", args.high)]:
            path = directory / f"{number:06d}.npz"
            if digest(path) != traces[key][number]["sha256"]:
                raise ValueError("Rendered frame changed or is incomplete")
            with np.load(path) as data:
                rgb = data["rgb"]
                rendered[key] = (rgb * 255).round().clip(0, 255).astype(np.uint8)
                row[key] = {
                    **image_metrics(reference, rendered[key]),
                    "render_sha256": digest(path),
                    "alpha_over_09": float(np.mean(data["alpha"] > 0.9)),
                }
        rows.append(row)
        display_width = min(640, reference.shape[1])
        display_height = round(reference.shape[0] * display_width / reference.shape[1])
        figure = Image.new(
            "RGB", (3 * display_width, display_height + 32), (24, 24, 24)
        )
        for column, (title, image) in enumerate(
            [
                (f"Captured frame {number}", reference),
                ("Training: 518 x 294; render: 1280 x 720", rendered["low"]),
                ("Training and render: 1280 x 720", rendered["high"]),
            ]
        ):
            ImageDraw.Draw(figure).text(
                (column * display_width + 6, 7), title, fill="white", font=font
            )
            figure.paste(
                Image.fromarray(image).resize(
                    (display_width, display_height), Image.Resampling.LANCZOS
                ),
                (column * display_width, 32),
            )
        figures.append(figure)
    sheet = Image.new(
        "RGB", (figures[0].width, sum(image.height for image in figures)), (24, 24, 24)
    )
    y = 0
    for image in figures:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(args.output / "comparison.jpg", quality=95)
    report = {
        "signatures": signatures,
        "rows": rows,
        "summary": {},
        "metric_accuracy_verified": False,
        "interpretation": "Both saved Gaussian models use identical calibrated 1280x720 cameras, reserved RGB and the same float32 renderer. Metrics use all pixels after identical eight-bit conversion. Images in the contact sheet are resized only for display. Appearance and opacity do not prove surface geometry or independent dimensional accuracy.",
    }
    for key in ["low", "high"]:
        report["summary"][key] = {
            metric: float(np.mean([row[key][metric] for row in rows]))
            for metric in ["full_image_psnr", "full_image_mae", "alpha_over_09"]
        }
    (args.output / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
