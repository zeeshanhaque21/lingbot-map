"""Float32 Gaussian rendering and experimental conditional depth on CPU/MPS.

Projection and alpha compositing follow Brush v0.3.0's Apache-2.0 shaders.
Depth moments are diagnostics, not a calibrated surface-accuracy guarantee.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from artifacts.research.experiments.mapanything_mac import digest


def read_gaussians(path):
    """Read the complete uncompressed float32 Gaussian PLY schema, by field name."""
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Expected PLY")
        properties, count, binary = [], None, False
        for _ in range(200):
            fields = stream.readline().decode("ascii").strip().split()
            if not fields:
                raise ValueError("Incomplete PLY header")
            if fields[0] == "format":
                binary = fields[1] == "binary_little_endian"
            elif fields[:2] == ["element", "vertex"]:
                count = int(fields[2])
            elif fields[0] == "element":
                raise ValueError("Only Gaussian vertex elements are supported")
            elif fields[0] == "property":
                if fields[1] not in ("float", "float32"):
                    raise ValueError("Expected float32 Gaussian properties")
                properties.append((fields[2], "<f4"))
            elif fields[0] == "end_header":
                break
        else:
            raise ValueError("Oversized PLY header")
        if not binary or count is None:
            raise ValueError("Expected little-endian Gaussian vertices")
        data = np.fromfile(stream, dtype=np.dtype(properties), count=count)
        if len(data) != count or stream.read(1):
            raise ValueError("PLY length does not match its schema")
    names = data.dtype.names
    rest = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name[7:]),
    )
    if rest != [f"f_rest_{i}" for i in range(len(rest))] or len(rest) not in (
        0,
        9,
        24,
        45,
    ):
        raise ValueError(
            "Require contiguous SH coefficients with degree zero through three"
        )

    def columns(keys):
        return np.column_stack([data[key] for key in keys]).astype(np.float32)

    dc = columns([f"f_dc_{i}" for i in range(3)])[:, None]
    coeffs = (
        np.concatenate(
            [dc, columns(rest).reshape(count, 3, -1).transpose(0, 2, 1)], axis=1
        )
        if rest
        else dc
    )
    result = {
        "means": columns(["x", "y", "z"]),
        "log_scales": columns([f"scale_{i}" for i in range(3)]),
        "quaternions": columns([f"rot_{i}" for i in range(4)]),
        "coefficients": coeffs,
        "raw_opacity": np.asarray(data["opacity"]).copy(),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Gaussian parameters contain nonfinite values")
    return result


def quaternion_matrices(quaternions):
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    q = quaternions / np.maximum(norms, 1e-30)
    w, x, y, z = q.T
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)


def sh_colors(coefficients, directions):
    x, y, z = directions.T
    basis = [np.full_like(x, 0.28209479177387814)]
    if coefficients.shape[1] >= 4:
        basis.extend(
            [-0.48860251190292 * y, 0.48860251190292 * z, -0.48860251190292 * x]
        )
    if coefficients.shape[1] >= 9:
        basis.extend(
            [
                1.092548430592079 * x * y,
                -1.092548430592079 * z * y,
                0.9461746957575601 * z * z - 0.3153915652525201,
                -1.092548430592079 * z * x,
                0.5462742152960395 * (x * x - y * y),
            ]
        )
    if coefficients.shape[1] >= 16:
        c = -2.285228997322329 * z * z + 0.4570457994644658
        basis.extend(
            [
                -0.5900435899266435 * y * (3 * x * x - y * y),
                2 * 1.445305721320277 * z * x * y,
                c * y,
                z * (1.865881662950577 * z * z - 1.119528997770346),
                c * x,
                1.445305721320277 * z * (x * x - y * y),
                -0.5900435899266435 * x * (x * x - 3 * y * y),
            ]
        )
    return np.maximum(
        0, np.sum(coefficients * np.stack(basis, axis=-1)[..., None], axis=1) + 0.5
    )


def project_gaussians(gaussians, pose, intrinsics, width, height):
    """Project anisotropic covariances and derive local conditional Z moments."""
    pose = np.asarray(pose, np.float32)
    extrinsic = np.linalg.inv(pose)
    rotation = extrinsic[:3, :3]
    means = gaussians["means"] @ rotation.T + extrinsic[:3, 3]
    opacity = 1 / (1 + np.exp(-np.clip(gaussians["raw_opacity"], -80, 80)))
    valid = (means[:, 2] >= 0.01) & (means[:, 2] <= 1e10) & (opacity >= 1 / 255)
    valid &= np.sum(gaussians["quaternions"] ** 2, axis=1) >= 1e-6
    indices = np.flatnonzero(valid)
    indices = indices[np.argsort(means[indices, 2], kind="stable")]
    means, opacity = means[indices], opacity[indices]
    matrices = quaternion_matrices(gaussians["quaternions"][indices])
    scaled = matrices * np.exp(gaussians["log_scales"][indices])[:, None, :]
    camera_scaled = rotation[None] @ scaled
    covariance = camera_scaled @ camera_scaled.transpose(0, 2, 1)
    focal = np.diag(intrinsics)[:2].astype(np.float32)
    center = np.asarray(intrinsics, np.float32)[:2, 2]
    uv = means[:, :2] / means[:, 2:3]
    clipped = np.clip(
        uv,
        (-0.15 * np.array([width, height]) - center) / focal,
        (1.15 * np.array([width, height]) - center) / focal,
    ).astype(np.float32)
    jacobian = np.zeros((len(indices), 2, 3), np.float32)
    jacobian[:, 0, 0] = focal[0] / means[:, 2]
    jacobian[:, 1, 1] = focal[1] / means[:, 2]
    jacobian[:, :, 2] = -clipped * focal / means[:, 2:3]
    cov2d = jacobian @ covariance @ jacobian.transpose(0, 2, 1)
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3
    inverse = np.linalg.inv(cov2d)
    xy = uv * focal + center
    extent = np.sqrt(
        2 * np.log(255 * opacity)[:, None] * np.diagonal(cov2d, axis1=1, axis2=2)
    )
    visible = ((xy + extent) > 0).all(axis=1) & ((xy - extent) < [width, height]).all(
        axis=1
    )
    # Conditioning a 3D Gaussian on its linearly projected image location.
    # This is not a trained surface constraint or an uncertainty calibration.
    cross = (covariance[:, 2:3, :] @ jacobian.transpose(0, 2, 1))[:, 0]
    slope = (cross[:, None, :] @ inverse)[:, 0]
    conditional_variance = np.maximum(
        0, covariance[:, 2, 2] - np.sum(slope * cross, axis=1)
    )
    view = gaussians["means"][indices] - pose[:3, 3]
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-30)
    color = sh_colors(gaussians["coefficients"][indices], view)
    packed = np.column_stack(
        [
            xy,
            inverse[:, 0, 0],
            inverse[:, 0, 1],
            inverse[:, 1, 1],
            opacity,
            color,
            means[:, 2],
            slope,
            conditional_variance,
            covariance[:, 2, 2],
        ]
    )
    return packed[visible].astype(np.float32), extent[visible].astype(np.float32)


def tile_indices(projected, extents, width, height):
    nx, ny = (width + 15) // 16, (height + 15) // 16
    lower = np.floor((projected[:, :2] - extents) / 16).astype(int)
    upper = np.ceil((projected[:, :2] + extents) / 16).astype(int)
    lower = np.clip(lower, [0, 0], [nx, ny])
    upper = np.clip(upper, [0, 0], [nx, ny])
    tiles = [[] for _ in range(nx * ny)]
    for index, (lo, hi) in enumerate(zip(lower, upper)):
        for y in range(lo[1], hi[1]):
            for x in range(lo[0], hi[0]):
                tiles[y * nx + x].append(index)
    return tiles


@torch.inference_mode()
def rasterize(
    projected, extents, width, height, device="cpu", batch_elements=4_000_000
):
    tiles = tile_indices(projected, extents, width, height)
    nx, ny = (width + 15) // 16, (height + 15) // 16
    packed = torch.as_tensor(projected, device=device)
    output = np.zeros((ny * 16, nx * 16, 13), np.float32)
    pending = [i for i, indices in enumerate(tiles) if indices]
    while pending:
        count, longest = 1, len(tiles[pending[0]])
        while count < min(16, len(pending)):
            next_longest = max(longest, len(tiles[pending[count]]))
            if (count + 1) * next_longest * 256 > batch_elements:
                break
            longest, count = next_longest, count + 1
        chosen, pending = pending[:count], pending[count:]
        indices = np.zeros((count, longest), np.int64)
        mask = np.zeros((count, longest), bool)
        for row, index in enumerate(chosen):
            indices[row, : len(tiles[index])] = tiles[index]
            mask[row, : len(tiles[index])] = True
        data = packed[torch.as_tensor(indices, device=device)]
        gy, gx = np.mgrid[:16, :16]
        pixels = np.stack(
            [
                np.stack(
                    [
                        gx.ravel() + (index % nx) * 16 + 0.5,
                        gy.ravel() + (index // nx) * 16 + 0.5,
                    ],
                    axis=-1,
                )
                for index in chosen
            ]
        ).astype(np.float32)
        delta = torch.as_tensor(pixels, device=device)[:, None] - data[:, :, None, :2]
        dx, dy = delta[..., 0], delta[..., 1]
        sigma = (
            0.5 * (data[:, :, None, 2] * dx * dx + data[:, :, None, 4] * dy * dy)
            + data[:, :, None, 3] * dx * dy
        )
        alpha = (data[:, :, None, 5] * torch.exp(-sigma)).clamp(max=0.999)
        alpha = torch.where(
            (sigma >= 0)
            & (alpha >= 1 / 255)
            & torch.as_tensor(mask, device=device)[:, :, None],
            alpha,
            0,
        )
        after = torch.cumprod(1 - alpha, dim=1)
        before = torch.cat([torch.ones_like(after[:, :1]), after[:, :-1]], dim=1)
        weights = alpha * before * (after > 1e-4)
        opacity = weights.sum(dim=1)
        denominator = opacity.clamp_min(1e-12)
        rgb = torch.einsum("bnp,bnc->bpc", weights, data[:, :, 6:9])
        z = data[:, :, None, 9]
        center_depth = (weights * z).sum(dim=1) / denominator
        center_variance = (
            (weights * (z - center_depth[:, None]) ** 2).sum(dim=1) / denominator
        ).clamp_min(0)
        conditional = z + dx * data[:, :, None, 10] + dy * data[:, :, None, 11]
        depth = (weights * conditional).sum(dim=1) / denominator
        variance = (
            weights * ((conditional - depth[:, None]) ** 2 + data[:, :, None, 12])
        ).sum(dim=1) / denominator
        component_variance = (weights * data[:, :, None, 13]).sum(dim=1) / denominator
        cumulative = weights.cumsum(dim=1)

        def quantile_value(
            values, fraction, cumulative=cumulative, opacity=opacity, weights=weights
        ):
            selected = (
                (cumulative >= opacity[:, None] * fraction)
                .to(torch.int32)
                .argmax(dim=1, keepdim=True)
            )
            value = values.expand_as(weights).gather(1, selected)[:, 0]
            return torch.where(opacity > 0, value, 0)

        median_center = quantile_value(z, 0.5)
        center_iqr = quantile_value(z, 0.75) - quantile_value(z, 0.25)
        median_depth = quantile_value(conditional, 0.5)
        median_sigma = quantile_value(data[:, :, None, 12].clamp_min(0).sqrt(), 0.5)
        result = (
            torch.cat(
                [
                    rgb,
                    opacity[..., None],
                    center_depth[..., None],
                    center_variance.sqrt()[..., None],
                    depth[..., None],
                    variance.clamp_min(0).sqrt()[..., None],
                    component_variance.clamp_min(0).sqrt()[..., None],
                    median_center[..., None],
                    center_iqr[..., None],
                    median_depth[..., None],
                    median_sigma[..., None],
                ],
                dim=-1,
            )
            .cpu()
            .numpy()
        )
        for row, index in enumerate(chosen):
            y, x = (index // nx) * 16, (index % nx) * 16
            output[y : y + 16, x : x + 16] = result[row].reshape(16, 16, 13)
    output = output[:height, :width]
    return {
        "rgb": output[..., :3],
        "alpha": output[..., 3],
        "center_depth": output[..., 4],
        "center_sigma": output[..., 5],
        "depth": output[..., 6],
        "sigma": output[..., 7],
        "component_sigma": output[..., 8],
        "median_center_depth": output[..., 9],
        "center_iqr": output[..., 10],
        "median_depth": output[..., 11],
        "median_sigma": output[..., 12],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--frames", help="Optional comma-separated original frame IDs")
    args = parser.parse_args()
    transform_path = args.dataset / f"transforms_{args.split}.json"
    signature = {
        "algorithm_sha256": digest(Path(__file__)),
        "ply_sha256": digest(args.ply),
        "transforms_sha256": digest(transform_path),
        "split": args.split,
        "device": args.device,
        "frames": args.frames,
    }
    signature_path = args.output / "signature.json"
    if args.output.exists():
        if (
            not signature_path.exists()
            or json.loads(signature_path.read_text()) != signature
        ):
            parser.error(
                "Output belongs to different inputs or code; use a new directory"
            )
    else:
        args.output.mkdir(parents=True)
        signature_path.write_text(json.dumps(signature, indent=2))
        (args.output / "producer.py").write_bytes(Path(__file__).read_bytes())
    gaussians = read_gaussians(args.ply)
    selected = set(map(int, args.frames.split(","))) if args.frames else None
    frames = json.loads(transform_path.read_text())["frames"]
    if selected is not None and selected - {
        int(Path(frame["file_path"]).stem) for frame in frames
    }:
        parser.error("Requested frame is absent from this split")
    trace_path = args.output / "trace.jsonl"
    previous = (
        {
            row["frame"]: row
            for row in (
                json.loads(line) for line in trace_path.read_text().splitlines()
            )
        }
        if trace_path.exists()
        else {}
    )
    for frame in frames:
        number = int(Path(frame["file_path"]).stem)
        if selected is not None and number not in selected:
            continue
        path = args.output / f"{number:06d}.npz"
        if path.exists():
            if number not in previous or digest(path) != previous[number]["sha256"]:
                raise ValueError(
                    "Incomplete or changed cached frame; preserve and use a new output"
                )
            continue
        start = time.monotonic()
        pose = np.asarray(frame["transform_matrix"]) @ np.diag([1, -1, -1, 1])
        intrinsic = np.array(
            [
                [frame["fl_x"], 0, frame["cx"]],
                [0, frame["fl_y"], frame["cy"]],
                [0, 0, 1],
            ]
        )
        projected, extents = project_gaussians(
            gaussians, pose, intrinsic, frame["w"], frame["h"]
        )
        rendered = rasterize(projected, extents, frame["w"], frame["h"], args.device)
        np.savez_compressed(path, **rendered)
        Image.fromarray(
            (rendered["rgb"] * 255).round().clip(0, 255).astype(np.uint8)
        ).save(args.output / f"{number:06d}.png")
        row = {
            "frame": number,
            "seconds": time.monotonic() - start,
            "projected_gaussians": len(projected),
            "alpha_over_09": float((rendered["alpha"] > 0.9).mean()),
            "sha256": digest(path),
            "metric_accuracy_verified": False,
        }
        with trace_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
