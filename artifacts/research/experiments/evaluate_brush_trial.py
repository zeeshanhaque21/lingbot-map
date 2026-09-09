"""Compare saved Brush evaluation renders with an unchanged TSDF baseline."""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

from artifacts.research.experiments.mapanything_mac import digest


def image_metrics(reference, rendered):
    if reference.shape != rendered.shape:
        raise ValueError("Images must have identical dimensions")
    difference = rendered.astype(np.float64) / 255 - reference.astype(np.float64) / 255
    return {
        "full_image_mae": float(np.abs(difference).mean()),
        "full_image_psnr": float(
            -10 * np.log10(max(float(np.mean(difference**2)), 1e-12))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing evaluations; use a new output directory")
    args.output.mkdir(parents=True)
    cameras = json.loads((args.baseline / "model/cameras.json").read_text())
    manifest = json.loads((args.baseline / "input.json").read_text())
    cloud_path = args.baseline / "model/observed-surfaces.ply"
    mesh = o3d.io.read_triangle_mesh(str(cloud_path))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    colors, triangles = np.asarray(mesh.vertex_colors), np.asarray(mesh.triangles)
    source_run = json.loads((args.run[0] / "run.json").read_text())
    dataset = Path(source_run["args"][1])
    references, baseline, baseline_metrics = {}, {}, []
    for camera in cameras:
        if not camera["held_out_from_fusion"]:
            continue
        original = manifest["frames"][camera["frame"]].get(
            "original_frame", camera["frame"]
        )
        filename = f"{original:06d}.png"
        reference = np.asarray(Image.open(dataset / "images" / filename).convert("RGB"))
        height, width = reference.shape[:2]
        rays = scene.create_rays_pinhole(
            np.asarray(camera["intrinsics"]),
            np.linalg.inv(camera["camera_to_world"]),
            width,
            height,
        )
        hit = scene.cast_rays(rays)
        visible = np.isfinite(hit["t_hit"].numpy())
        primitive = hit["primitive_ids"].numpy()[visible]
        uv = hit["primitive_uvs"].numpy()[visible]
        weights = np.column_stack([1 - uv.sum(1), uv])
        rendered = np.zeros_like(reference)
        rendered[visible] = (
            (255 * np.sum(colors[triangles[primitive]] * weights[..., None], axis=1))
            .clip(0, 255)
            .astype(np.uint8)
        )
        references[filename], baseline[filename] = reference, rendered
        Image.fromarray(rendered).save(args.output / f"baseline-{filename}")
        baseline_metrics.append(
            {
                "frame": original,
                "coverage": float(visible.mean()),
                **image_metrics(reference, rendered),
            }
        )
    results = {
        "baseline": {
            "source": str(args.baseline.resolve()),
            "mesh_sha256": digest(cloud_path),
            "views": baseline_metrics,
        },
        "runs": [],
        "metric_accuracy_verified": False,
        "ready_for_verified_property_listing": False,
        "interpretation": "Full-image RGB metrics include missing pixels, rendered black for the unchanged TSDF. Brush evaluation RGB is excluded from splat optimization, but prior geometry/camera estimation used all frames. These appearance measurements do not establish geometry or metric accuracy.",
    }
    for run in args.run:
        record = json.loads((run / "run.json").read_text())
        run_dataset = Path(record["args"][1])
        validation = json.loads((run_dataset / "transforms_val.json").read_text())
        expected = {Path(frame["file_path"]).name for frame in validation["frames"]}
        if expected != set(references):
            raise ValueError("Every run must use the same reserved views")
        for filename in expected:
            if not np.array_equal(
                references[filename],
                np.asarray(
                    Image.open(run_dataset / "images" / filename).convert("RGB")
                ),
            ):
                raise ValueError(f"Reference image differs: {filename}")
        checkpoints = []
        for folder in sorted(run.glob("eval_*"), key=lambda item: int(item.name[5:])):
            if {path.name for path in folder.glob("*.png")} != expected:
                raise ValueError(f"Incomplete evaluation: {folder}")
            views = [
                {
                    "frame": int(name[:-4]),
                    **image_metrics(
                        references[name],
                        np.asarray(Image.open(folder / name).convert("RGB")),
                    ),
                }
                for name in sorted(expected)
            ]
            checkpoints.append(
                {
                    "step": int(folder.name[5:]),
                    "views": views,
                    "mean_full_image_psnr": float(
                        np.mean([v["full_image_psnr"] for v in views])
                    ),
                    "mean_full_image_mae": float(
                        np.mean([v["full_image_mae"] for v in views])
                    ),
                }
            )
        if not checkpoints:
            raise ValueError(f"No saved evaluation images: {run}")
        final = checkpoints[-1]["step"]
        exports = []
        for path in sorted(run.glob("export_*.ply")):
            with path.open("rb") as stream:
                header = []
                for _ in range(100):
                    line = stream.readline().decode("ascii").strip()
                    header.append(line)
                    if line == "end_header":
                        break
                else:
                    raise ValueError("Unexpected PLY header")
            exports.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                    "splats": int(
                        next(
                            line.split()[2]
                            for line in header
                            if line.startswith("element vertex ")
                        )
                    ),
                }
            )
        results["runs"].append(
            {
                "source": str(run.resolve()),
                "configuration": record,
                "completion": json.loads((run / "completion.json").read_text()),
                "dataset_provenance": json.loads(
                    (run_dataset / "provenance.json").read_text()
                ),
                "checkpoints": checkpoints,
                "exports": exports,
            }
        )
        first = next(iter(references.values()))
        height, width = first.shape[:2]
        sheet = Image.new(
            "RGB", (width * 3, (height + 28) * len(expected)), (24, 24, 24)
        )
        draw = ImageDraw.Draw(sheet)
        for row, name in enumerate(sorted(expected)):
            y = row * (height + 28)
            rendered = np.asarray(
                Image.open(run / f"eval_{final}" / name).convert("RGB")
            )
            for column, (title, array) in enumerate(
                [
                    (f"Captured frame {int(name[:-4])}", references[name]),
                    ("Unchanged TSDF; missing = black", baseline[name]),
                    (f"Brush at {final} steps", rendered),
                ]
            ):
                draw.text((column * width + 6, y + 7), title, fill="white")
                sheet.paste(Image.fromarray(array), (column * width, y + 28))
        sheet.save(args.output / f"{run.name}-comparison.jpg", quality=94)
    results["baseline"]["mean_full_image_psnr"] = float(
        np.mean([v["full_image_psnr"] for v in baseline_metrics])
    )
    results["baseline"]["mean_full_image_mae"] = float(
        np.mean([v["full_image_mae"] for v in baseline_metrics])
    )
    (args.output / "results.json").write_text(json.dumps(results, indent=2))
    print(
        json.dumps(
            {
                "baseline_psnr": results["baseline"]["mean_full_image_psnr"],
                "runs": [
                    {"source": r["source"], "last": r["checkpoints"][-1]}
                    for r in results["runs"]
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
