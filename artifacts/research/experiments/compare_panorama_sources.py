"""Compare a station panorama with photos that were excluded from mesh fusion.

Camera and mesh errors affect this diagnostic; it does not certify architecture.
The same visibility mask is used for the unchanged baseline and candidate.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from lingbot_map.reconstruction.io import digest, write_json
from lingbot_map.reconstruction.rendering import SurfaceRenderer
from lingbot_map.reconstruction.tour import captured_rgb


def source_coordinates(renderer, camera, station_pose, width, height, panorama_size):
    """Backproject source pixels, rejecting surfaces occluded from the station."""
    k = np.asarray(camera["intrinsics"], dtype=float).copy()
    k[:2, 2] += 0.5 - camera.get("pixel_center_offset", 0.0)
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        k, np.linalg.inv(camera["camera_to_world"]), width, height
    )
    depth = renderer.scene.cast_rays(rays)["t_hit"].numpy()
    rays = rays.numpy()
    hit = np.isfinite(depth)
    points = rays[..., :3] + rays[..., 3:] * np.where(hit, depth, 0)[..., None]
    station_pose = np.asarray(station_pose)
    vectors = points - station_pose[:3, 3]
    distance = np.linalg.norm(vectors, axis=-1)
    directions = vectors / np.maximum(distance[..., None], 1e-8)
    station_rays = np.concatenate(
        [np.broadcast_to(station_pose[:3, 3], directions.shape), directions], axis=-1
    ).astype(np.float32)
    station_depth = renderer.scene.cast_rays(o3d.core.Tensor(station_rays))[
        "t_hit"
    ].numpy()
    visible = (
        hit
        & (distance > 1e-6)
        & np.isfinite(station_depth)
        & (np.abs(station_depth - distance) <= np.maximum(distance * 0.001, 1e-5))
    )
    # Exclude one pixel along occlusion boundaries before comparing colors.
    visible = cv2.erode(visible.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    local = directions @ station_pose[:3, :3]
    pano_width, pano_height = panorama_size
    x = (
        np.arctan2(local[..., 0], local[..., 2]) / (2 * np.pi) + 0.5
    ) * pano_width - 0.5
    y = (0.5 + np.arcsin(np.clip(local[..., 1], -1, 1)) / np.pi) * pano_height - 0.5
    return x.astype(np.float32), y.astype(np.float32), visible


def sample_panorama(panorama, x, y):
    """Wrap longitude while clamping latitude; never wrap across opposite poles."""
    padded = np.concatenate([panorama[:, -1:], panorama, panorama[:, :1]], axis=1)
    return cv2.remap(
        padded,
        (x % panorama.shape[1] + 1).astype(np.float32),
        np.clip(y, 0, panorama.shape[0] - 1).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def color_error(image, reference, mask):
    if not mask.any():
        return None
    delta = (image[mask].astype(np.float32) - reference[mask]) / 255
    return {
        "absolute_rgb_error": float(np.abs(delta).mean()),
        "psnr": float(-10 * np.log10(max(float(np.mean(delta**2)), 1e-12))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tour", type=Path, required=True)
    parser.add_argument("--station", default="000")
    parser.add_argument("--panorama", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous comparisons; choose a fresh --output directory")
    tour = json.loads((args.tour / "tour.json").read_text())
    stations = json.loads((args.tour / "stations.json").read_text())
    station = next((s for s in stations if s["id"] == args.station), None)
    if station is None:
        parser.error("Unknown station; inspect the tour's stations.json")
    source = Path(tour["source"])
    cameras_path = source / "model/cameras.json"
    if digest(cameras_path) != tour["source_cameras_sha256"]:
        parser.error("Source cameras changed since tour export")
    cameras = json.loads(cameras_path.read_text())
    selected = [
        c
        for c in cameras
        if c["held_out_from_fusion"] and c["frame"] != station["frame"]
    ]
    if not selected:
        parser.error("No non-anchor photos excluded from fusion are available")
    candidate = np.asarray(Image.open(args.panorama).convert("RGB"))
    baseline = np.asarray(Image.open(args.baseline).convert("RGB"))
    if (
        candidate.shape != baseline.shape
        or candidate.shape[1] != 2 * candidate.shape[0]
    ):
        parser.error("Use matching 2:1 candidate and baseline panoramas")
    renderer = SurfaceRenderer.from_file(args.tour / "render-mesh.ply")
    args.output.mkdir(parents=True)
    records = []
    for camera in selected:
        reference = captured_rgb(source, camera)
        height, width = reference.shape[:2]
        x, y, mask = source_coordinates(
            renderer,
            camera,
            station["camera_to_world"],
            width,
            height,
            (candidate.shape[1], candidate.shape[0]),
        )
        raw_view = sample_panorama(baseline, x, y)
        candidate_view = sample_panorama(candidate, x, y)
        record = {
            "frame": camera["frame"],
            "shared_visible_pixels": int(mask.sum()),
            "shared_visible_fraction": float(mask.mean()),
            "baseline": color_error(raw_view, reference, mask),
            "candidate": color_error(candidate_view, reference, mask),
        }
        records.append(record)
        sheet = Image.new("RGB", (width * 3, height + 32), (25, 30, 34))
        draw = ImageDraw.Draw(sheet)
        for i, (label, pixels) in enumerate(
            [
                (f"Captured frame {camera['frame']}", reference),
                ("Unchanged panorama, shared surfaces", raw_view),
                ("Candidate panorama, same surfaces", candidate_view),
            ]
        ):
            draw.text((i * width + 8, 10), label, fill="white")
            pixels = pixels.copy()
            if i:
                pixels[~mask] = 24
            sheet.paste(Image.fromarray(pixels), (i * width, 32))
        sheet.save(args.output / f"frame-{camera['frame']:06d}.jpg", quality=94)
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            args.output / f"mask-{camera['frame']:06d}.png"
        )
        write_json(args.output / "frames.json", records)
        print(f"frame: {camera['frame']}, shared_pixels: {int(mask.sum())}", flush=True)
    report = {
        "station": args.station,
        "source_photos": len(records),
        "photos_with_shared_surfaces": sum(
            r["shared_visible_pixels"] > 0 for r in records
        ),
        "candidate_sha256": digest(args.panorama),
        "baseline_sha256": digest(args.baseline),
        "mesh_sha256": digest(args.tour / "render-mesh.ply"),
        "cameras_sha256": digest(cameras_path),
        "visibility_tolerance_relative": 0.001,
        "visibility_tolerance_absolute": 1e-5,
        "visibility_boundary_erosion_pixels": 1,
        "records": records,
        "interpretation": "These photos were excluded from mesh fusion and are not FixAnything anchors. They still contributed to camera estimation and learned reconstruction context. Reprojection uses the estimated mesh and cameras. Color errors diagnose changed appearance on shared surfaces; they do not certify structural fidelity or unseen areas.",
        "architectural_fidelity_verified": False,
    }
    write_json(args.output / "comparison.json", report)
    print(f"report: {args.output / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
