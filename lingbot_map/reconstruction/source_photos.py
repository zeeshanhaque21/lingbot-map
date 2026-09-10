"""Project captured photographs onto observed surfaces with depth checks."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .geometry import preprocessing_transform
from .io import digest


def load_photos(source, maximum=24):
    """Load a bounded, deterministic sample, reserving fusion holdouts for review."""
    source = Path(source)
    cameras = json.loads((source / "model/cameras.json").read_text())
    manifest = json.loads((source / "input.json").read_text())
    candidates = [c for c in cameras if not c.get("held_out_from_fusion", False)]
    if maximum < 1 or not candidates:
        raise ValueError("At least one non-held-out source photograph is required")
    selected = np.linspace(
        0, len(candidates) - 1, min(maximum, len(candidates)), dtype=int
    )
    cameras = [candidates[i] for i in selected]
    windows = sorted((source / "windows").glob("*.npz"))
    photos = []
    for window in sorted({c["window"] for c in cameras}):
        archive_hash = digest(windows[window])
        with np.load(windows[window]) as archive:
            data = {key: archive[key] for key in ("frame_ids", "depth", "confidence")}
            for camera in cameras:
                if camera["window"] != window:
                    continue
                frame = camera["frame"]
                index = int(np.flatnonzero(data["frame_ids"] == frame)[0])
                path = source / manifest["frames"][frame]["file"]
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert("RGB")).copy()
                depth = data["depth"][index].copy()
                confidence = data["confidence"][index]
                depth[~np.isfinite(depth) | (confidence <= 2.5)] = 0
                height, width = depth.shape
                transform = preprocessing_transform(
                    rgb.shape[1], rgb.shape[0], width, height
                )
                photos.append(
                    {
                        "frame": frame,
                        "rgb": rgb,
                        "depth": depth,
                        "intrinsics": np.asarray(camera["intrinsics"]),
                        "photo_intrinsics": np.linalg.inv(transform)
                        @ camera["intrinsics"],
                        "world_to_camera": np.linalg.inv(camera["camera_to_world"]),
                        "path": str(path.resolve()),
                        "sha256": digest(path),
                        "depth_archive_sha256": archive_hash,
                    }
                )
    return sorted(photos, key=lambda p: p["frame"])


def project_photos(world, visible, base_rgb, target_center, photos, tolerance=0.03):
    """Copy the best supported source pixel, without inventing surface coverage.

    Source depth rejects occluded surfaces and disagreements with the mesh.
    The return map identifies the actual photograph used at every replaced pixel.
    Reprojected pixels remain degraded conditioning, never clean-frame anchors.
    """
    output = base_rgb.copy()
    provenance = np.full(visible.shape, -1, np.int32)
    valid = visible & np.isfinite(world).all(axis=-1)
    points = world[valid]
    if not len(points):
        return output, provenance
    selected_rgb = output[valid].copy()
    selected_frame = provenance[valid].copy()
    best = np.full(len(points), np.inf)
    target_distance = np.linalg.norm(points - target_center, axis=1)
    for photo in photos:
        ext = photo["world_to_camera"]
        camera = points @ ext[:3, :3].T + ext[:3, 3]
        projected = camera @ photo["intrinsics"].T
        xy = projected[:, :2] / np.maximum(projected[:, 2:], 1e-8)
        height, width = photo["depth"].shape
        inside = (
            (camera[:, 2] > 0)
            & (xy[:, 0] >= 1)
            & (xy[:, 0] < width - 2)
            & (xy[:, 1] >= 1)
            & (xy[:, 1] < height - 2)
        )
        indices = np.flatnonzero(inside)
        if not len(indices):
            continue
        # OpenCV remap dimensions must remain below SHRT_MAX.
        for start in range(0, len(indices), 32766):
            batch = indices[start : start + 32766]
            coordinates = xy[batch].astype("float32")
            observed = cv2.remap(
                photo["depth"],
                coordinates[:, 0, None],
                coordinates[:, 1, None],
                cv2.INTER_NEAREST,
            )[:, 0]
            residual = np.abs(camera[batch, 2] - observed) / np.maximum(observed, 1e-8)
            center = -ext[:3, :3].T @ ext[:3, 3]
            baseline = np.linalg.norm(center - target_center) / np.maximum(
                target_distance[batch], 1e-8
            )
            edge = np.minimum.reduce(
                [
                    coordinates[:, 0] / width,
                    (width - 1 - coordinates[:, 0]) / width,
                    coordinates[:, 1] / height,
                    (height - 1 - coordinates[:, 1]) / height,
                ]
            )
            score = baseline + residual + 0.02 / np.maximum(edge, 0.01)
            replace = (observed > 0) & (residual <= tolerance) & (score < best[batch])
            batch, score = batch[replace], score[replace]
            if not len(batch):
                continue
            photo_pixels = camera[batch] @ photo["photo_intrinsics"].T
            uv = (photo_pixels[:, :2] / photo_pixels[:, 2:]).astype("float32")
            sampled = cv2.remap(
                photo["rgb"],
                uv[:, 0, None],
                uv[:, 1, None],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )[:, 0]
            selected_rgb[batch] = sampled
            selected_frame[batch] = photo["frame"]
            best[batch] = score
    output[valid], provenance[valid] = selected_rgb, selected_frame
    return output, provenance
