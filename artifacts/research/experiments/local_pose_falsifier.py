"""Paired local TSDF experiment; intentionally independent of pipeline helpers."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw


def filtered_depth(data, i):
    depth, confidence = data["depth"][i], data["confidence"][i]
    valid = np.isfinite(depth) & (depth > 0)
    valid &= confidence > np.quantile(confidence, 0.25)
    kernel = np.ones((3, 3), np.uint8)
    valid &= cv2.dilate(depth, kernel) - cv2.erode(depth, kernel) < 0.08 * depth
    y, x = np.indices(depth.shape)
    pixels = np.stack([x, y, np.ones_like(x)], -1)
    camera = (pixels @ np.linalg.inv(data["intrinsics"][i]).T) * depth[..., None]
    e = data["extrinsics"][i]
    world = (camera - e[:, 3]) @ e[:, :3]
    support = np.zeros(depth.shape, bool)
    for j in (i - 3, i - 1, i + 1, i + 3):
        if j < 0 or j >= len(data["depth"]):
            continue
        e = data["extrinsics"][j]
        other = world @ e[:, :3].T + e[:, 3]
        projected = other @ data["intrinsics"][j].T
        uv = (projected[..., :2] / np.maximum(projected[..., 2:], 1e-8)).astype(np.float32)
        observed = cv2.remap(data["depth"][j], uv[..., 0], uv[..., 1], cv2.INTER_NEAREST)
        conf = cv2.remap(data["confidence"][j], uv[..., 0], uv[..., 1], cv2.INTER_NEAREST)
        support |= ((other[..., 2] > 0) & (observed > 0)
                    & (conf > np.quantile(data["confidence"][j], 0.25))
                    & (np.abs(other[..., 2] - observed) < 0.03 * observed))
    return np.where(valid & support, depth, 0).astype(np.float32)


def reconstruct(data, target, radius, destination, label):
    started = time.monotonic()
    center = int(np.flatnonzero(data["frame_ids"] == target)[0])
    selected = [i for i, frame in enumerate(data["frame_ids"])
                if abs(int(frame) - target) <= radius and int(frame) % 10 != 5]
    voxel = float(np.median(data["depth"]) / 180)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=voxel * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for i in selected:
        depth = filtered_depth(data, i)
        h, w = depth.shape
        k = data["intrinsics"][i]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(data["rgb"][i])),
            o3d.geometry.Image(depth), depth_scale=1, depth_trunc=float(depth.max() + 1),
            convert_rgb_to_intensity=False,
        )
        e = np.eye(4)
        e[:3] = data["extrinsics"][i]
        volume.integrate(rgbd, intrinsic, e)
    mesh = volume.extract_triangle_mesh()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    e = np.eye(4)
    e[:3] = data["extrinsics"][center]
    reference = data["rgb"][center]
    h, w = reference.shape[:2]
    rays = scene.create_rays_pinhole(data["intrinsics"][center].astype(float), e, w, h)
    hits = scene.cast_rays(rays)
    visible = np.isfinite(hits["t_hit"].numpy())
    rendered_depth = hits["t_hit"].numpy() * (rays.numpy()[..., 3:] @ e[2, :3])
    expected = data["depth"][center]
    confidence = data["confidence"][center]
    valid = (confidence >= np.quantile(confidence, 0.25)) & (expected > 0)
    relative_error = np.abs(rendered_depth - expected) / np.maximum(expected, 1e-8)
    primitive = hits["primitive_ids"].numpy()[visible]
    uv = hits["primitive_uvs"].numpy()[visible]
    weights = np.column_stack([1 - uv.sum(1), uv])
    colors = np.asarray(mesh.vertex_colors)[np.asarray(mesh.triangles)[primitive]]
    rendered = np.full(reference.shape, 24, np.uint8)
    rendered[visible] = (255 * np.sum(colors * weights[..., None], 1)).clip(0, 255).astype(np.uint8)
    result = {
        "target": target, "arm": label, "frames_fused": len(selected),
        "coverage": float(visible.mean()),
        "supported_depth_fraction": float(np.sum(valid & visible & (relative_error < 0.05)) / valid.sum()),
        "rgb_mae_visible": float(np.abs(rendered.astype(float) - reference)[visible].mean() / 255),
        "median_depth_error": float(np.median(relative_error[visible & valid])),
        "triangles": len(mesh.triangles), "seconds": time.monotonic() - started,
    }
    row = Image.new("RGB", (w * 2, h + 28), (24, 24, 24))
    row.paste(Image.fromarray(reference), (0, 28))
    row.paste(Image.fromarray(rendered), (w, 28))
    ImageDraw.Draw(row).text((8, 7), f"{target}: {label}; support {result['supported_depth_fraction']:.3f}", fill="white")
    row.save(destination / f"{target}-{label}.jpg", quality=93)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--registered", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / "results.jsonl"
    done = {(r["target"], r["arm"]) for r in map(json.loads, log.read_text().splitlines())} if log.exists() else set()
    for start, target in [(936, 945), (504, 515)]:
        for source, name in [(args.registered, "registered"), (args.native, "native")]:
            data = dict(np.load(source / "windows" / f"{start:06d}.npz"))
            for radius, span in [(9999, "window"), (8, "neighbors")]:
                label = f"{name}-{span}"
                if (target, label) in done:
                    continue
                result = reconstruct(data, target, radius, args.output, label)
                with log.open("a") as f:
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
