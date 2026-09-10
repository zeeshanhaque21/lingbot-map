"""Compare saved model predictions and processing stages on one local sample.

This deliberately implements projection and filtering independently of the app.
No model inference or source artifact mutation occurs.
The source photographs used for evaluation are excluded from geometry fusion,
but participated in learned inference, so these are not independent ground truth.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw


def world_points(depth, intrinsic, extrinsic):
    y, x = np.indices(depth.shape)
    pixels = np.stack([x, y, np.ones_like(x)], axis=-1)
    camera = (pixels @ np.linalg.inv(intrinsic).T) * depth[..., None]
    return (camera - extrinsic[:, 3]) @ extrinsic[:, :3]


def filter_depth(data, index):
    depth, confidence = data["depth"][index], data["confidence"][index]
    valid = (
        np.isfinite(depth) & (depth > 0) & (confidence > np.quantile(confidence, 0.25))
    )
    kernel = np.ones((3, 3), np.uint8)
    valid &= cv2.dilate(depth, kernel) - cv2.erode(depth, kernel) < 0.08 * depth
    points = world_points(depth, data["intrinsics"][index], data["extrinsics"][index])
    supported = np.zeros(depth.shape, bool)
    for other in [index - 3, index - 1, index + 1, index + 3]:
        if not 0 <= other < len(data["depth"]):
            continue
        ext, intrinsic = data["extrinsics"][other], data["intrinsics"][other]
        camera = points @ ext[:, :3].T + ext[:, 3]
        pixels = camera @ intrinsic.T
        coordinates = (pixels[..., :2] / np.maximum(pixels[..., 2:], 1e-8)).astype(
            "float32"
        )
        observed = cv2.remap(
            data["depth"][other],
            coordinates[..., 0],
            coordinates[..., 1],
            cv2.INTER_NEAREST,
        )
        conf = cv2.remap(
            data["confidence"][other],
            coordinates[..., 0],
            coordinates[..., 1],
            cv2.INTER_NEAREST,
        )
        supported |= (
            (camera[..., 2] > 0)
            & (observed > 0)
            & (conf > np.quantile(data["confidence"][other], 0.25))
            & (np.abs(camera[..., 2] - observed) < 0.03 * observed)
        )
    return np.where(valid & supported, depth, 0).astype("float32")


def render_points(points, colors, intrinsic, extrinsic, shape, radius):
    height, width = shape
    camera = points @ extrinsic[:, :3].T + extrinsic[:, 3]
    front = camera[:, 2] > 1e-5
    camera, colors = camera[front], colors[front]
    projected = camera @ intrinsic.T
    xy = np.rint(projected[:, :2] / projected[:, 2:]).astype("int32")
    keep = (
        (xy[:, 0] >= -radius)
        & (xy[:, 0] < width + radius)
        & (xy[:, 1] >= -radius)
        & (xy[:, 1] < height + radius)
    )
    xy, camera, colors = xy[keep], camera[keep], colors[keep]
    buffer = np.full(height * width, np.inf)
    # Far to near yields an explicit nearest-point winner per pixel.
    order = np.argsort(camera[:, 2])[::-1]
    xy, z, colors = xy[order], camera[order, 2], colors[order]
    rgb = np.full((height * width, 3), 24, np.uint8)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            u, v = xy[:, 0] + dx, xy[:, 1] + dy
            good = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            pixel = v[good] * width + u[good]
            indices = np.flatnonzero(good)
            _, reverse = np.unique(pixel[::-1], return_index=True)
            indices = indices[len(pixel) - 1 - reverse]
            pixel = (xy[indices, 1] + dy) * width + xy[indices, 0] + dx
            closer = z[indices] < buffer[pixel]
            pixel, indices = pixel[closer], indices[closer]
            buffer[pixel], rgb[pixel] = z[indices], colors[indices]
    return rgb.reshape(height, width, 3), np.isfinite(buffer).reshape(height, width)


def render_mesh(mesh, intrinsic, extrinsic, shape):
    height, width = shape
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    y, x = np.indices(shape)
    directions = (
        np.stack([x, y, np.ones_like(x)], -1)
        @ np.linalg.inv(intrinsic).T
        @ extrinsic[:, :3]
    )
    center = -extrinsic[:, :3].T @ extrinsic[:, 3]
    rays = np.concatenate(
        [np.broadcast_to(center, directions.shape), directions], axis=-1
    ).astype("float32")
    hit = scene.cast_rays(o3d.core.Tensor(rays))
    valid = np.isfinite(hit["t_hit"].numpy())
    faces = np.asarray(mesh.triangles)[hit["primitive_ids"].numpy()[valid]]
    uv = hit["primitive_uvs"].numpy()[valid]
    weight = np.column_stack([1 - uv.sum(axis=1), uv])
    rgb = np.full((height, width, 3), 24, np.uint8)
    rgb[valid] = (
        np.rint(
            (np.asarray(mesh.vertex_colors)[faces] * weight[..., None]).sum(axis=1)
            * 255
        )
        .clip(0, 255)
        .astype("uint8")
    )
    return rgb, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw_path = Path("reconstructions/examples/loop/windows/000000.npz")
    registered_path = Path(
        "reconstructions/examples/loop-closure-v1/final/windows/000000.npz"
    )
    raw, registered = dict(np.load(raw_path)), dict(np.load(registered_path))
    targets = [5, 35, 65]
    training = [i for i in range(72) if i % 10 != 5]
    depth = [filter_depth(registered, i) for i in training]
    # Match production: the reference statistic spans the first saved window.
    voxel = float(np.median(registered["depth"]) / 180)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=voxel * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    shape = raw["depth"][0].shape
    height, width = shape
    for i, d in zip(training, depth):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(registered["rgb"][i].copy()),
            o3d.geometry.Image(d),
            depth_scale=1,
            depth_trunc=1000,
            convert_rgb_to_intensity=False,
        )
        k = registered["intrinsics"][i]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        )
        ext = np.eye(4)
        ext[:3] = registered["extrinsics"][i]
        volume.integrate(rgbd, intrinsic, ext)
    mesh = volume.extract_triangle_mesh()
    simplified = mesh.simplify_quadric_decimation(150000)
    print(
        json.dumps(
            {
                "event": "fused",
                "triangles": len(mesh.triangles),
                "simplified": len(simplified.triangles),
            }
        ),
        flush=True,
    )
    cloud = {}
    for name, data, filtered in [
        ("raw_points", raw, False),
        ("registered_points", registered, False),
        ("filtered_points", registered, True),
    ]:
        xyz, color = [], []
        for j, i in enumerate(training):
            d = depth[j] if filtered else data["depth"][i]
            mask = (d > 0) & np.isfinite(d)
            if not filtered:
                mask &= data["confidence"][i] > 2.5
            # Same two-pixel sampling in every point representation.
            mask[1::2] = False
            mask[:, 1::2] = False
            xyz.append(
                world_points(d, data["intrinsics"][i], data["extrinsics"][i])[mask]
            )
            color.append(data["rgb"][i][mask])
        cloud[name] = np.concatenate(xyz), np.concatenate(color)
    stages = [
        "raw_points",
        "registered_points",
        "filtered_points",
        "tsdf",
        "simplified",
    ]
    metrics, images = [], {}
    for i in targets:
        images[(i, "source")] = raw["rgb"][i]
        for stage in stages:
            data = raw if stage == "raw_points" else registered
            if stage in cloud:
                rgb, visible = render_points(
                    *cloud[stage],
                    data["intrinsics"][i],
                    data["extrinsics"][i],
                    shape,
                    radius=1,
                )
            else:
                rgb, visible = render_mesh(
                    mesh if stage == "tsdf" else simplified,
                    data["intrinsics"][i],
                    data["extrinsics"][i],
                    shape,
                )
            images[(i, stage)] = rgb
            difference = np.abs(rgb.astype(float) - raw["rgb"][i]) / 255
            metrics.append(
                {
                    "frame": i,
                    "stage": stage,
                    "coverage": float(visible.mean()),
                    "whole_rgb_l1": float(difference.mean()),
                    "visible_rgb_l1": float(difference[visible].mean()),
                }
            )
            Image.fromarray(rgb).save(args.output / f"{i:03d}-{stage}.png")
        Image.fromarray(raw["rgb"][i]).save(args.output / f"{i:03d}-source.png")
        print(json.dumps({"event": "rendered", "frame": i}), flush=True)
    stages = ["source"] + stages
    sheet = Image.new("RGB", (width * 3, (height + 28) * 6), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for row, stage in enumerate(stages):
        for column, i in enumerate(targets):
            x, y = column * width, row * (height + 28)
            sheet.paste(Image.fromarray(images[(i, stage)]), (x, y + 28))
            draw.text((x + 5, y + 5), f"{stage} / frame {i}", fill="white")
    sheet.save(args.output / "comparison.jpg", quality=94)
    report = {
        "input_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (raw_path, registered_path)
        },
        "inference_window_frames": 96,
        "sample_frames": 72,
        "fused_frames": len(training),
        "evaluation_frames": targets,
        "point_pixel_stride": 2,
        "point_splat_radius_pixels": 1,
        "raw_confidence_threshold": 2.5,
        "voxel_model_units": voxel,
        "tsdf_triangles": len(mesh.triangles),
        "simplified_triangles": len(simplified.triangles),
        "accepted_depth_fraction": float(np.mean([np.mean(d > 0) for d in depth])),
        "metrics": metrics,
        "limitations": [
            "Same input, different representations; point rasterizer approximates the official display, not an exact reproduction.",
            "Evaluation RGB participated in learned inference but is excluded from fused geometry.",
            "Only first 72 frames of one 96-frame window; global drift and window joins are not exercised.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
