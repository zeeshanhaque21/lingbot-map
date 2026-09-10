"""Audit shared-center cameras and render one station against its captured panorama."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lingbot_map.reconstruction.rendering import SurfaceRenderer
from lingbot_map.reconstruction.tour import panorama_rays


def camera_poses(data):
    poses = np.tile(np.eye(4), (len(data["extrinsics"]), 1, 1))
    poses[:, :3] = data["extrinsics"]
    return np.linalg.inv(poses)


def rig_audit(data, records):
    poses = camera_poses(data)
    rows = []
    for capture_id in sorted({r["capture_id"] for r in records}):
        group = [r for r in records if r["capture_id"] == capture_id]
        reference = next(r for r in group if r["yaw_degrees"] == 0)
        anchor = poses[reference["frame"]]
        for row in group:
            if row is reference:
                continue
            pose = poses[row["frame"]]
            orientation = pose[:3, :3] @ np.asarray(row["camera_to_panorama_rotation"]).T
            error = anchor[:3, :3].T @ orientation
            angle = np.degrees(np.arccos(np.clip((np.trace(error) - 1) / 2, -1, 1)))
            rows.append({
                "capture_id": capture_id,
                "yaw_degrees": row["yaw_degrees"],
                "rotation_error_degrees": float(angle),
                "center_error_model_units": float(np.linalg.norm(pose[:3, 3] - anchor[:3, 3])),
            })
    forward = [r["frame"] for r in records if r["yaw_degrees"] == 0]
    depth_scale = float(np.median(data["depth"][forward]))
    maximum_spread = [
        max(r["center_error_model_units"] for r in rows if r["capture_id"] == capture_id)
        for capture_id in sorted({r["capture_id"] for r in rows})
    ]
    expected_focal = np.array([np.asarray(r["intrinsics"])[[0, 1], [0, 1]] for r in records])
    focal_error = np.abs(data["intrinsics"][:, [0, 1], [0, 1]] / expected_focal - 1)
    return {
        "capture_count": len(maximum_spread),
        "rotation_pair_count": len(rows),
        "median_rotation_error_degrees": float(np.median([r["rotation_error_degrees"] for r in rows])),
        "maximum_rotation_error_degrees": max(r["rotation_error_degrees"] for r in rows),
        "median_maximum_center_spread_model_units": float(np.median(maximum_spread)),
        "reference_median_depth_model_units": depth_scale,
        "median_maximum_center_spread_over_scene_depth": float(np.median(maximum_spread) / depth_scale),
        "median_absolute_relative_focal_error": float(np.median(focal_error)),
        "ideal_shared_center_error": 0.0,
        "ideal_relative_rotation_error_degrees": 0.0,
        "interpretation": "Known projection-rig constraints, not independent measurements of building dimensions",
        "pairs": rows,
    }


def package_captured_station(pilot, evaluation, report):
    """Reuse the property viewer with observed panorama pixels at one camera station."""
    root = evaluation / "captured-tour"
    station = root / "stations/000"
    station.mkdir(parents=True, exist_ok=False)
    source = np.array(Image.open(evaluation / "captured-panorama.png").convert("RGB"))
    mask = np.array(Image.open(evaluation / "captured-mask.png")) > 0
    source[~mask] = 24
    Image.fromarray(source).save(station / "panorama.png")
    shutil.copy2(evaluation / "captured-mask.png", station / "observed-mask.png")
    frame = report["capture_id"]
    shutil.copy2(pilot / "single/images" / f"{frame:06d}.png", station / "reference.png")
    shutil.copy2(pilot / "omni/run/model/property.glb", root / "property.glb")
    metadata = json.loads((pilot / "projection.json").read_text())
    stations = [{
        "id": "000",
        "frame": frame,
        "source_video_frame": report["source_frame"],
        "timestamp_seconds": report["timestamp_seconds"],
        "camera_to_world": report["arms"]["omni"]["station_pose"],
        "intrinsics": metadata["arms"]["single"][frame]["intrinsics"],
        "reference_size": metadata["perspective_size"],
        "panorama": "stations/000/panorama.png",
        "observed_mask": "stations/000/observed-mask.png",
        "appearance": "captured_360",
        "generated": False,
        "panorama_sha256": hashlib.sha256((station / "panorama.png").read_bytes()).hexdigest(),
    }]
    tour = {
        "source": str(pilot),
        "station_count": 1,
        "navigation_edges": [],
        "loop_constraints_verified": False,
        "metric_accuracy_verified": False,
        "mesh_fidelity_passed": False,
        "fixanything_refinement": "not_run",
        "panorama_convention": "Captured equirectangular RGB, center +Z, right +X, top -Y; invalid source regions masked",
        "interpretation": "The captured panorama is real source footage; the associated rough mesh failed fidelity screening.",
    }
    for name, value in [("tour.json", tour), ("stations.json", stations)]:
        with (root / name).open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
    return "captured-tour"


def evaluate(pilot, output, capture_id=15):
    pilot, output = Path(pilot).resolve(), Path(output).resolve()
    metadata = json.loads((pilot / "projection.json").read_text())
    if not 0 <= capture_id < len(metadata["source_frames"]):
        raise ValueError("Capture index is outside the prepared pilot")
    if output.exists():
        raise ValueError("Choose a new evaluation directory")
    data = {arm: dict(np.load(pilot / arm / "run/windows/000000.npz")) for arm in ["single", "omni"]}
    output.mkdir(parents=True)
    source_frame = metadata["source_frames"][capture_id]
    source = np.array(Image.open(pilot / "panoramas" / f"{source_frame:06d}.png").convert("RGB"))
    height, width = source.shape[:2]
    mask = np.array(Image.open(pilot / "panoramas/source-mask.png")) > 0
    weights = np.broadcast_to(np.cos(((np.arange(height) + 0.5) / height - 0.5) * np.pi)[:, None], mask.shape)
    denominator = float(weights[mask].sum())
    Image.fromarray(source).save(output / "captured-panorama.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(output / "captured-mask.png")
    report = {
        "source_frame": source_frame,
        "timestamp_seconds": metadata["source_timestamps_seconds"][capture_id],
        "capture_id": capture_id,
        "single_run_per_arm": True,
        "metric_accuracy_verified": False,
        "loop_closure_verified": False,
        "source_panorama_is_generated": False,
        "source_mask_canvas_fraction": float(mask.mean()),
        "source_mask_sphere_fraction": float(weights[mask].sum() / weights.sum()),
        "do_nothing_captured_panorama": {"rgb_l1_on_valid_source": 0.0, "valid_source_coverage": 1.0},
        "rig": rig_audit(data["omni"], metadata["arms"]["omni"]),
        "unchanged_forward_predictions": {
            field: bool(np.array_equal(data["single"][field], data["omni"][field][:len(data["single"][field])]))
            for field in ["rgb", "depth", "confidence", "intrinsics", "extrinsics"]
        },
        "arms": {},
        "interpretation": "Captured source pixels are the appearance baseline; same-source mesh checks do not establish independent 3D accuracy. Other directions from this capture can participate in fusion.",
    }
    sheets = [("Captured panorama - invalid source regions shaded", source.copy())]
    sheets[0][1][~mask] = (32, 32, 32)
    shared_forward = {r["frame"] for r in metadata["arms"]["single"] if r["frame"] % 10 == 5}
    for arm in ["single", "omni"]:
        run = pilot / arm / "run"
        record = next(r for r in metadata["arms"][arm] if r["capture_id"] == capture_id and r["yaw_degrees"] == 0)
        pose = camera_poses(data[arm])[record["frame"]]
        # Remove only floating-point orthogonality noise before constructing rays.
        u, _, vt = np.linalg.svd(pose[:3, :3])
        pose[:3, :3] = u @ vt
        renderer = SurfaceRenderer.from_file(run / "model/property.glb", glb=True)
        rendered, depth, visible = renderer.render_rays(panorama_rays(pose, width, height))
        visible_source = visible & mask
        difference = np.abs(rendered.astype(np.float64) - source) / 255
        Image.fromarray(rendered).save(output / f"{arm}-mesh-panorama.png")
        Image.fromarray((visible_source * 255).astype(np.uint8)).save(output / f"{arm}-mesh-mask.png")
        np.save(output / f"{arm}-radial-depth.npy", np.where(visible, depth, 0))
        model = json.loads((run / "model/validation.json").read_text())
        validation = json.loads((run / "model/web-render-validation.json").read_text())
        shared = [r for r in validation["views"] if r["frame"] in shared_forward]
        coverage = float(weights[visible_source].sum() / denominator)
        report["arms"][arm] = {
            "frames": len(data[arm]["frame_ids"]),
            "triangles": model["triangles"],
            "web_triangles": model["web_triangles"],
            "view_consistency_gate": validation["view_consistency_gate"],
            "validation_view_count": len(validation["views"]),
            "all_reserved_median_coverage": validation["median_rendered_coverage"],
            "same_forward_reserved_views": shared,
            "station_mesh_coverage_of_valid_source_sphere": coverage,
            "station_rgb_l1_visible": float(difference[visible_source].mean()) if visible_source.any() else None,
            "station_rgb_l1_full_valid_source": float((difference.mean(-1)[mask] * weights[mask]).sum() / denominator),
            "station_pose": pose.tolist(),
            "station_forward_view_withheld_from_fusion": record["frame"] % 10 == 5,
        }
        sheets.append((f"{arm.title()} direction mesh - {coverage:.1%} of valid source sphere", rendered))
    preview_width = 1280
    preview_height = round(height * preview_width / width)
    sheet = Image.new("RGB", (preview_width, (preview_height + 38) * len(sheets)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    for index, (label, pixels) in enumerate(sheets):
        y = index * (preview_height + 38)
        draw.text((12, y + 8), label, fill="white", font=font)
        resized = Image.fromarray(pixels).resize((preview_width, preview_height), Image.Resampling.LANCZOS)
        sheet.paste(resized, (0, y + 38))
    sheet.save(output / "station-comparison.jpg", quality=92)
    report["captured_tour"] = package_captured_station(pilot, output, report)
    with (output / "result.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capture", type=int, default=15)
    args = parser.parse_args()
    if args.pilot is None or args.output is None:
        print("description: Audit one completed AIST pilot and its captured panorama station")
        print('help: "Provide --pilot <pilot-directory> --output <new-directory>; optional --capture 15"')
        return 0 if args.pilot is None and args.output is None else 2
    try:
        result = evaluate(args.pilot, args.output, args.capture)
    except (ValueError, OSError, RuntimeError) as error:
        print("error: " + json.dumps(str(error)))
        print('help: "Complete both reconstruction arms and use a new --output directory"')
        return 1
    print("status: inspected")
    for arm, item in result["arms"].items():
        print(f"{arm}_passed: " + str(item["view_consistency_gate"]).lower())
        print(f"{arm}_station_coverage: {item['station_mesh_coverage_of_valid_source_sphere']:.6f}")
    print("output: " + json.dumps(str(args.output.resolve())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
