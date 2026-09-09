"""Package a native scene and audit every reserved captured view without invented depth."""

import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image, ImageDraw, ImageFont

from artifacts.research.experiments.evaluate_openmvs_trial import glb_document
from lingbot_map.reconstruction.colmap_io import read_model
from lingbot_map.reconstruction.io import digest
from lingbot_map.reconstruction.rendering import SurfaceRenderer

AXIS = np.diag([1.0, -1.0, -1.0, 1.0])


def frame_number(frame):
    stem = Path(frame["file_path"]).stem
    if len(stem) != 6 or not stem.isdigit():
        raise ValueError("Expected a captured six-digit frame filename")
    number = int(stem)
    if frame.get("original_frame", number) != number:
        raise ValueError("Frame identifier differs from its captured filename")
    return number


def native_resources(trial):
    completion = json.loads((trial / "completion.json").read_text())
    source = trial / "property.glb"
    if not completion["complete"] or digest(source) != completion["property_sha256"]:
        raise ValueError("Native output is incomplete or changed")
    resources = {"property.glb": digest(source)}
    document = glb_document(source)
    for item in document.get("images", []) + document.get("buffers", []):
        uri = item.get("uri")
        if not uri or uri.startswith("data:"):
            continue
        path = (trial / unquote(uri)).resolve()
        if not path.is_relative_to(trial.resolve()):
            raise ValueError("Native resource escapes the trial directory")
        resources[uri] = digest(path)
    return resources


def package_scene(source, destination):
    """Keep each geometry, material and node; prove the exported scene round trip."""
    scene = trimesh.load_scene(source, process=False)
    if not scene.graph.nodes_geometry:
        raise ValueError("Native scene has no triangle geometry")
    packed = scene.copy()
    packed.apply_transform(AXIS)
    packed.export(destination)
    document = glb_document(destination)
    if any(
        "uri" in item
        for item in document.get("images", []) + document.get("buffers", [])
    ):
        raise ValueError("Export still needs external resources")
    recovered = trimesh.load_scene(destination, process=False)
    if set(scene.graph.nodes_geometry) != set(recovered.graph.nodes_geometry):
        raise ValueError("Packaging changed scene nodes")
    rows = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        after_transform, after_name = recovered.graph[node]
        before, after = scene.geometry[name], recovered.geometry[after_name]
        if not np.allclose(after_transform, AXIS @ transform, atol=1e-7, rtol=0):
            raise ValueError("Packaging changed node placement")
        if not np.array_equal(before.faces, after.faces) or not np.allclose(
            before.vertices, after.vertices, atol=1e-7, rtol=0
        ):
            raise ValueError("Packaging changed geometry")
        texture = getattr(before.visual.material, "baseColorTexture", None)
        after_texture = getattr(after.visual.material, "baseColorTexture", None)
        if texture is None or after_texture is None:
            raise ValueError("Expected native textured geometry")
        if not np.array_equal(np.asarray(texture), np.asarray(after_texture)):
            raise ValueError("Packaging changed texture pixels")
        if not np.allclose(before.visual.uv, after.visual.uv, atol=1e-7, rtol=0):
            raise ValueError("Packaging changed texture coordinates")
        rows.append(
            {
                "node": node,
                "triangles": len(before.faces),
                "texture_size": list(texture.size),
            }
        )
    return rows


def sparse_consistency(renderer, image, points):
    """Camera-estimation tracks are consistency evidence, not independent truth."""
    camera, extrinsic = image["camera"], image["extrinsics"]
    k = camera["intrinsics"]
    c2w = np.linalg.inv(np.vstack([extrinsic, [0, 0, 0, 1]]))
    pixels, reference_z = [], []
    for x, y, identifier in image["observations"]:
        point = points.get(int(identifier))
        if point is None or point["track_length"] < 3:
            continue
        position = extrinsic[:, :3] @ point["xyz"] + extrinsic[:, 3]
        if position[2] <= 0:
            continue
        projected = k @ position
        if np.linalg.norm(projected[:2] / projected[2] - [x, y]) > 2:
            continue
        if not (0.5 <= x < camera["width"] - 0.5 and 0.5 <= y < camera["height"] - 0.5):
            continue
        pixels.append([x, y, 1])
        reference_z.append(position[2])
    if not pixels:
        return {"observations": 0}
    directions = np.asarray(pixels) @ np.linalg.inv(k).T @ c2w[:3, :3].T
    rays = np.column_stack([np.broadcast_to(c2w[:3, 3], directions.shape), directions])
    hit = renderer.scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))[
        "t_hit"
    ].numpy()
    visible = np.isfinite(hit)
    predicted_z = hit * (directions @ extrinsic[2, :3])
    error = np.abs(predicted_z - reference_z) / reference_z
    return {
        "observations": len(pixels),
        "coverage": float(visible.mean()),
        "within_5percent_all_tracks": float(np.mean(visible & (error < 0.05))),
        "median_relative_error_covered": float(np.median(error[visible]))
        if visible.any()
        else None,
        "p95_relative_error_covered": float(np.quantile(error[visible], 0.95))
        if visible.any()
        else None,
    }


def render_metrics(renderer, image, captured):
    camera = image["camera"]
    extrinsic = np.vstack([image["extrinsics"], [0, 0, 0, 1]])
    rendered, _, visible = renderer.render(
        camera["intrinsics"], extrinsic, camera["width"], camera["height"]
    )
    error = np.abs(rendered.astype(np.float32) - captured.astype(np.float32)) / 255
    black_missing = rendered.copy()
    black_missing[~visible] = 0
    mse = np.mean(((black_missing.astype(np.float32) - captured) / 255) ** 2)
    return {
        "rendered_coverage": float(visible.mean()),
        "absolute_rgb_error_visible": float(error[visible].mean())
        if visible.any()
        else None,
        "absolute_rgb_error_all_pixels_missing_black": float(
            np.abs(black_missing.astype(np.float32) - captured).mean() / 255
        ),
        "psnr_all_pixels_missing_black": float(-10 * np.log10(max(float(mse), 1e-12))),
    }, rendered


def write_comparison(path, number, captured, rendered, baseline=None):
    width, height = captured.shape[1], captured.shape[0]
    panels = [("Captured", captured), ("Native mesh", rendered)]
    if baseline is not None:
        panels.append(("Unchanged local mesh", baseline))
    font = ImageFont.load_default(size=max(16, width // 40))
    header = max(40, width // 30)
    sheet = Image.new("RGB", (width * len(panels), height + header), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for index, (label, rgb) in enumerate(panels):
        sheet.paste(Image.fromarray(rgb), (index * width, header))
        draw.text(
            (index * width + 12, 5), f"Frame {number}: {label}", font=font, fill="white"
        )
    sheet.save(path, quality=92)


def checked_provenance(trial, dataset, colmap_hashes):
    provenance = json.loads((dataset / "provenance.json").read_text())
    run = json.loads((trial / "run.json").read_text())
    if digest(dataset / "provenance.json") != run["dataset_provenance_sha256"]:
        raise ValueError("Dataset differs from native run provenance")
    if provenance["source_colmap_sha256"] != colmap_hashes:
        raise ValueError(
            "Native scene and evaluation cameras have different source models"
        )
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--captured-dataset", type=Path, required=True)
    parser.add_argument("--colmap-text", type=Path, required=True)
    parser.add_argument("--baseline-trial", type=Path)
    parser.add_argument("--baseline-dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing results; choose a new output directory")
    if bool(args.baseline_trial) != bool(args.baseline_dataset):
        parser.error("Provide both baseline trial and baseline dataset")
    colmap_hashes = {
        name: digest(args.colmap_text / name)
        for name in ("cameras.txt", "images.txt", "points3D.txt")
    }
    provenance = checked_provenance(args.trial, args.dataset, colmap_hashes)
    capture_provenance = json.loads(
        (args.captured_dataset / "provenance.json").read_text()
    )
    if capture_provenance["colmap_sha256"] != colmap_hashes:
        raise ValueError("Captured dataset calibration changed")
    frames = json.loads((args.captured_dataset / "transforms_val.json").read_text())[
        "frames"
    ]
    frame_ids = [frame_number(frame) for frame in frames]
    if (
        not frame_ids
        or len(set(frame_ids)) != len(frame_ids)
        or set(frame_ids) != {int(x) for x in provenance["reserved_images_excluded"]}
    ):
        raise ValueError("Reserved image split differs from dense reconstruction")
    resources = native_resources(args.trial)
    baseline, baseline_ids, baseline_resources = None, set(), None
    if args.baseline_trial:
        baseline_provenance = checked_provenance(
            args.baseline_trial, args.baseline_dataset, colmap_hashes
        )
        baseline_ids = {
            int(x) for x in baseline_provenance["reserved_images_excluded"]
        } & set(frame_ids)
        if not baseline_ids:
            raise ValueError("Baseline has no common reserved views")
        baseline_resources = native_resources(args.baseline_trial)
        baseline = SurfaceRenderer.from_file(
            args.baseline_trial / "property.glb", glb=True, app_axes=False
        )
    model = args.output / "model"
    model.mkdir(parents=True)
    (args.output / "views").mkdir()
    shutil.copy2(__file__, args.output / "producer.py")
    repository = Path(__file__).resolve().parents[3]
    sources = {
        name: digest(repository / name)
        for name in (
            "artifacts/research/experiments/evaluate_openmvs_capture.py",
            "artifacts/research/experiments/evaluate_openmvs_trial.py",
            "lingbot_map/reconstruction/rendering.py",
            "lingbot_map/reconstruction/colmap_io.py",
            "lingbot_map/reconstruction/io.py",
        )
    }
    parts = package_scene(args.trial / "property.glb", model / "property.glb")
    renderer = SurfaceRenderer.from_file(model / "property.glb", glb=True)
    images, points = read_model(args.colmap_text)
    rows = []
    with (args.output / "views.jsonl").open("x") as trace:
        for frame in frames:
            number = frame_number(frame)
            image = images[f"{number:06d}.jpg"]
            path = (args.captured_dataset / frame["file_path"]).resolve()
            if not path.is_relative_to(args.captured_dataset.resolve()):
                raise ValueError("Captured image escapes its dataset")
            if digest(path) != capture_provenance["image_sha256"][frame["file_path"]]:
                raise ValueError("Captured pixels changed")
            camera = image["camera"]
            expected_k = np.array(
                [
                    [frame["fl_x"], 0, frame["cx"]],
                    [0, frame["fl_y"], frame["cy"]],
                    [0, 0, 1],
                ]
            )
            expected_c2w = (
                np.linalg.inv(np.vstack([image["extrinsics"], [0, 0, 0, 1]])) @ AXIS
            )
            if not np.allclose(
                expected_k, camera["intrinsics"], atol=1e-7, rtol=0
            ) or not np.allclose(
                frame["transform_matrix"], expected_c2w, atol=1e-7, rtol=0
            ):
                raise ValueError("Reserved calibration differs from source COLMAP")
            captured = np.asarray(Image.open(path).convert("RGB"))
            if captured.shape[:2] != (camera["height"], camera["width"]):
                raise ValueError("Captured pixels have different camera dimensions")
            if (frame["h"], frame["w"]) != captured.shape[:2]:
                raise ValueError("Captured transform dimensions differ from pixels")
            metrics, rendered = render_metrics(renderer, image, captured)
            row = {
                "frame": number,
                "image_sha256": digest(path),
                **metrics,
                "sparse_consistency": sparse_consistency(renderer, image, points),
            }
            baseline_rgb = None
            if number in baseline_ids:
                baseline_metrics, baseline_rgb = render_metrics(
                    baseline, image, captured
                )
                row["unchanged_local_baseline"] = {
                    **baseline_metrics,
                    "sparse_consistency": sparse_consistency(baseline, image, points),
                }
            write_comparison(
                args.output / "views" / f"{number:06d}.jpg",
                number,
                captured,
                rendered,
                baseline_rgb,
            )
            trace.write(json.dumps(row) + "\n")
            trace.flush()
            rows.append(row)
            print(json.dumps({"frame": number, **metrics}), flush=True)
    if any(digest(repository / name) != value for name, value in sources.items()):
        raise ValueError("Evaluation sources changed during this run")
    result = {
        "producer_sources_sha256": sources,
        "source_resources_sha256": resources,
        "property_sha256": digest(model / "property.glb"),
        "property_bytes": (model / "property.glb").stat().st_size,
        "parts": parts,
        "triangles": sum(part["triangles"] for part in parts),
        "colmap_sha256": colmap_hashes,
        "capture_provenance_sha256": digest(args.captured_dataset / "provenance.json"),
        "dataset_provenance_sha256": digest(args.dataset / "provenance.json"),
        "reserved_transforms_sha256": digest(
            args.captured_dataset / "transforms_val.json"
        ),
        "baseline_resources_sha256": baseline_resources,
        "paired_baseline_frames": sorted(baseline_ids),
        "reserved_views": len(rows),
        "median_rendered_coverage": float(
            np.median([row["rendered_coverage"] for row in rows])
        ),
        "minimum_rendered_coverage": min(row["rendered_coverage"] for row in rows),
        "median_absolute_rgb_error_visible": float(
            np.median(
                [
                    row["absolute_rgb_error_visible"]
                    for row in rows
                    if row["absolute_rgb_error_visible"] is not None
                ]
            )
        ),
        "mean_psnr_all_pixels_missing_black": float(
            np.mean([row["psnr_all_pixels_missing_black"] for row in rows])
        ),
        "views": rows,
        "metric_accuracy_verified": False,
        "ready_for_verified_property_listing": False,
        "interpretation": "All reserved captured RGB views, with unchanged COLMAP cameras and no per-view alignment. Reserved images were excluded from dense stereo and texturing but participated in camera estimation. Sparse tracks measure consistency with that same calibration, not independent geometry accuracy. No generated or model-derived depth is treated as truth. The unchanged local mesh baseline is compared only on its common reserved frames. All native materials are embedded and geometry, texture pixels, UVs and node transforms pass a complete export round trip. Scale and building connectivity remain unverified.",
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "triangles",
                    "property_bytes",
                    "reserved_views",
                    "median_rendered_coverage",
                    "mean_psnr_all_pixels_missing_black",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
