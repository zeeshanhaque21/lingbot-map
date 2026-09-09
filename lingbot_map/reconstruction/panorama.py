"""Render a spherical camera sweep and stitch known camera rotations."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .io import digest, write_json
from .rendering import SurfaceRenderer
from .tour import panorama_rays


def sweep_rotations():
    """61 smooth neighboring views, both poles, identical first/last pose."""
    angles = [(0, 0), (0, -15), (0, -30), (0, -60), (0, -90)]
    for pitch, count in [(-60, 8), (-30, 12), (0, 12), (30, 12), (60, 8)]:
        # Reverse each ring to keep transition yaw increments short.
        sign = 1 if pitch in (-60, 0, 60) else -1
        angles.extend((sign * i * 360 / count, pitch) for i in range(count))
    angles.extend([(0, 90), (0, 60), (0, 30), (0, 0)])
    rotations = []
    for yaw, pitch in np.deg2rad(angles):
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        rotations.append(
            np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        )
    assert len(rotations) == 61
    return rotations


def export_sweep(tour, station_id, output, width=832, height=480):
    tour, output = Path(tour).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new sweep output directory")
    if width % 16 or height % 16 or min(width, height) < 64:
        raise ValueError("Sweep dimensions must be multiples of 16 and at least 64")
    stations = json.loads((tour / "stations.json").read_text())
    station = next((s for s in stations if s["id"] == station_id), None)
    if station is None:
        raise ValueError(f"Unknown station: {station_id}")
    source_pose = np.asarray(station["camera_to_world"])
    intrinsics = np.asarray(station["intrinsics"], dtype=float)
    sx, sy = np.array([width, height]) / station["reference_size"]
    intrinsics[0] *= sx
    intrinsics[1] *= sy
    intrinsics[0, 2] += (sx - 1) / 2
    intrinsics[1, 2] += (sy - 1) / 2
    renderer = SurfaceRenderer.from_file(tour / "render-mesh.ply")
    output.mkdir(parents=True)
    (output / "frames").mkdir()
    (output / "raw-frames").mkdir()
    reference_path = tour / "stations" / station_id / "reference.png"
    reference = (
        Image.open(reference_path)
        .convert("RGB")
        .resize((width, height), Image.Resampling.LANCZOS)
    )
    frames = []
    for i, rotation in enumerate(sweep_rotations()):
        pose = source_pose.copy()
        pose[:3, :3] = source_pose[:3, :3] @ rotation
        ray_intrinsics = intrinsics.copy()
        ray_intrinsics[:2, 2] += (
            0.5  # Open3D rays sample centers at (u + 0.5, v + 0.5).
        )
        rgb, _, visible = renderer.render(
            ray_intrinsics, np.linalg.inv(pose), width, height
        )
        raw = Image.fromarray(rgb)
        raw.save(output / "raw-frames" / f"{i:03d}.png")
        (reference if i in (0, 60) else raw).save(output / "frames" / f"{i:03d}.png")
        frames.append(
            {
                "index": i,
                "rotation": rotation.tolist(),
                "camera_to_world": pose.tolist(),
                "observed_fraction": float(visible.mean()),
                "captured_anchor": i in (0, 60),
            }
        )
        write_json(output / "frames.json", frames)
    manifest = {
        "station": station_id,
        "tour": str(tour),
        "width": width,
        "height": height,
        "intrinsics": intrinsics.tolist(),
        "frames": frames,
        "clean_frame_indices": [0, 60],
        "mesh_sha256": digest(tour / "render-mesh.ply"),
        "reference_sha256": digest(reference_path),
        "generated": False,
    }
    write_json(output / "sweep.json", manifest)
    return manifest


def stitch_sweep(sweep, images, output, width=2048, generated=False):
    """Reproject known rotations with center weighting; never estimate new geometry."""
    sweep, images, output = Path(sweep), Path(images), Path(output)
    if output.exists():
        raise ValueError("Choose a new stitched panorama output directory")
    config = json.loads((sweep / "sweep.json").read_text())
    paths = sorted(images.glob("*.png"))
    if len(paths) != len(config["frames"]):
        raise ValueError("Stitching requires one PNG for every sweep camera")
    if width < 64 or width % 2:
        raise ValueError("Panorama width must be even and at least 64")
    directions = panorama_rays(np.eye(4), width, width // 2)[..., 3:]
    sums = np.zeros((*directions.shape[:2], 3), np.float32)
    weights = np.zeros(directions.shape[:2], np.float32)
    squares = sums.copy()
    k = np.asarray(config["intrinsics"])
    for path, frame in zip(paths, config["frames"], strict=True):
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
        if rgb.shape[:2] != (config["height"], config["width"]):
            raise ValueError(f"Refined frame dimensions changed: {path}")
        camera_rays = directions @ np.asarray(frame["rotation"])
        projected = camera_rays @ k.T
        xy = projected[..., :2] / np.maximum(projected[..., 2:], 1e-8)
        valid = (
            (camera_rays[..., 2] > 0)
            & (xy[..., 0] >= 0)
            & (xy[..., 0] <= config["width"] - 1)
            & (xy[..., 1] >= 0)
            & (xy[..., 1] <= config["height"] - 1)
        )
        # Taper to zero at image boundaries, including the wraparound longitude.
        edge = np.minimum.reduce(
            [
                xy[..., 0],
                config["width"] - 1 - xy[..., 0],
                xy[..., 1],
                config["height"] - 1 - xy[..., 1],
            ]
        )
        weight = np.where(valid, np.clip(edge / 24, 0, 1) * camera_rays[..., 2] ** 4, 0)
        sample = cv2.remap(
            rgb,
            xy[..., 0].astype(np.float32),
            xy[..., 1].astype(np.float32),
            cv2.INTER_LINEAR,
        )
        sums += sample * weight[..., None]
        squares += sample**2 * weight[..., None]
        weights += weight
    covered = weights > 1e-6
    mean = sums / np.maximum(weights[..., None], 1e-6)
    variance = np.maximum(squares / np.maximum(weights[..., None], 1e-6) - mean**2, 0)
    output.mkdir(parents=True)
    Image.fromarray((np.clip(mean, 0, 1) * 255).round().astype(np.uint8)).save(
        output / "panorama.png"
    )
    Image.fromarray(covered.astype(np.uint8) * 255).save(output / "coverage.png")
    report = {
        "camera_coverage_fraction": float(covered.mean()),
        "mean_overlap_rgb_std": float(np.sqrt(variance[covered]).mean()),
        "wrap_seam_rgb_difference": float(np.abs(mean[:, 0] - mean[:, -1]).mean()),
        "generated": generated,
        "source_sweep": str(sweep.resolve()),
        "source_images": str(images.resolve()),
        "panorama_sha256": digest(output / "panorama.png"),
        "interpretation": "Coverage measures sweep directions, not observed building surfaces. Overlap disagreement and seam values require comparison with raw renders.",
    }
    write_json(output / "stitch.json", report)
    return report
