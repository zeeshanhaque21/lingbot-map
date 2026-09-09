"""Camera geometry and robust registration, independent of inference backend."""

import cv2
import numpy as np


def preprocessing_transform(width, height, processed_width, processed_height):
    """Match the upstream resize-to-width and center-crop preprocessing."""
    if processed_width % 14 or processed_height % 14:
        raise ValueError(
            "Processed image dimensions must be multiples of the model's 14-pixel patch size"
        )
    resized_height = round(height * processed_width / width / 14) * 14
    crop_top = max(0, (resized_height - processed_height) // 2)
    return np.array(
        [
            [processed_width / width, 0, 0],
            [0, resized_height / height, -crop_top],
            [0, 0, 1.0],
        ]
    )


def unproject(depth, intrinsics, extrinsics):
    """Camera z-depth to world XYZ, using OpenCV world-to-camera matrices."""
    y, x = np.indices(depth.shape)
    rays = np.stack([x, y, np.ones_like(x)], axis=-1) @ np.linalg.inv(intrinsics).T
    camera = rays * depth[..., None]
    return (camera - extrinsics[:, 3]) @ extrinsics[:, :3]


def transform_points(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def similarity(source, target, weights=None):
    """Weighted Umeyama fit: target = scale * rotation * source + translation."""
    if len(source) < 4:
        raise ValueError("At least four correspondences are required")
    weights = np.ones(len(source)) if weights is None else weights
    weights = weights / weights.sum()
    mx, my = weights @ source, weights @ target
    x, y = source - mx, target - my
    u, singular, vt = np.linalg.svd((y * weights[:, None]).T @ x)
    correction = np.ones(3)
    correction[-1] = np.sign(np.linalg.det(u @ vt))
    rotation = (u * correction) @ vt
    variance = np.sum(weights[:, None] * x * x)
    if variance < 1e-12:
        raise ValueError("Degenerate registration geometry")
    scale = np.sum(singular * correction) / variance
    result = np.eye(4)
    result[:3, :3] = scale * rotation
    result[:3, 3] = my - scale * rotation @ mx
    return result


def robust_similarity(source, target):
    transform = similarity(source, target)
    for _ in range(12):
        residual = np.linalg.norm(transform_points(source, transform) - target, axis=1)
        cutoff = max(float(np.median(residual)) * 2.5, 1e-6)
        weights = 1 / (1 + (residual / cutoff) ** 2)
        transform = similarity(source, target, weights)
    return transform, np.linalg.norm(
        transform_points(source, transform) - target, axis=1
    )


def align_overlap(previous, current):
    common, old_ids, new_ids = np.intersect1d(
        previous["frame_ids"], current["frame_ids"], return_indices=True
    )
    if len(common) < 3:
        raise ValueError("Windows have fewer than three shared frames")
    source, target = [], []
    for old, new in zip(old_ids, new_ids):
        a = unproject(
            previous["depth"][old],
            previous["intrinsics"][old],
            previous["extrinsics"][old],
        )
        b = unproject(
            current["depth"][new],
            current["intrinsics"][new],
            current["extrinsics"][new],
        )
        valid = previous["confidence"][old] > np.quantile(
            previous["confidence"][old], 0.5
        )
        valid &= current["confidence"][new] > np.quantile(
            current["confidence"][new], 0.5
        )
        valid &= np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
        source.append(b[::8, ::8][valid[::8, ::8]])
        target.append(a[::8, ::8][valid[::8, ::8]])
    source, target = np.concatenate(source), np.concatenate(target)
    transform, errors = robust_similarity(source, target)
    reference = float(np.median(previous["depth"][old_ids]))
    stats = {
        "shared_frames": len(common),
        "correspondences": len(source),
        "scale": float(np.cbrt(np.linalg.det(transform[:3, :3]))),
        "median_relative_error": float(np.median(errors) / reference),
        "p95_relative_error": float(np.quantile(errors, 0.95) / reference),
    }
    if not 0.5 < stats["scale"] < 2 or stats["median_relative_error"] > 0.08:
        raise ValueError(f"Unreliable overlap registration: {stats}")
    return transform, stats


def supported_depth(index, data, relative_tolerance=0.03):
    """Keep depth supported by another view; reject uncertain pixels and depth edges."""
    depth = data["depth"][index]
    confidence = data["confidence"][index]
    valid = (
        np.isfinite(depth) & (depth > 0) & (confidence > np.quantile(confidence, 0.25))
    )
    high = cv2.dilate(depth, np.ones((3, 3), np.uint8))
    low = cv2.erode(depth, np.ones((3, 3), np.uint8))
    valid &= (high - low) < 0.08 * np.maximum(depth, 1e-6)
    xyz = unproject(depth, data["intrinsics"][index], data["extrinsics"][index])
    support = np.zeros(depth.shape, np.uint8)
    tested, errors = 0, []
    for neighbor in (index - 3, index - 1, index + 1, index + 3):
        if not 0 <= neighbor < len(data["depth"]):
            continue
        if "registered" in data and not data["registered"][neighbor]:
            continue
        extrinsic, intrinsic = (
            data["extrinsics"][neighbor],
            data["intrinsics"][neighbor],
        )
        camera = xyz @ extrinsic[:, :3].T + extrinsic[:, 3]
        pixels = camera @ intrinsic.T
        denominator = np.maximum(pixels[..., 2], 1e-8)
        u, v = pixels[..., 0] / denominator, pixels[..., 1] / denominator
        observed = cv2.remap(
            data["depth"][neighbor],
            u.astype(np.float32),
            v.astype(np.float32),
            cv2.INTER_NEAREST,
        )
        observed_conf = cv2.remap(
            data["confidence"][neighbor],
            u.astype(np.float32),
            v.astype(np.float32),
            cv2.INTER_NEAREST,
        )
        projected_valid = (
            valid
            & (camera[..., 2] > 0)
            & (observed > 0)
            & (observed_conf > np.quantile(data["confidence"][neighbor], 0.25))
        )
        discrepancy = np.abs(camera[..., 2] - observed) / np.maximum(observed, 1e-8)
        # Only geometrically visible points enter the residual diagnostic.
        visible = projected_valid & (
            camera[..., 2] <= observed * (1 + relative_tolerance)
        )
        errors.extend(discrepancy[visible][::100].tolist())
        support += projected_valid & (discrepancy < relative_tolerance)
        tested += int(projected_valid.sum())
    accepted = valid & (support >= 1)
    stats = {
        "frame": int(data["frame_ids"][index]),
        "pixels": int(depth.size),
        "candidate_pixels": int(valid.sum()),
        "accepted_pixels": int(accepted.sum()),
        "projected_comparisons": tested,
        "median_visible_relative_residual": float(np.median(errors))
        if errors
        else None,
    }
    return np.where(accepted, depth, 0).astype(np.float32), stats
