"""Test bounded multiview photometric depth correction with fixed cameras.

Search locally scaled depth hypotheses using patch-normalized image agreement.
This is an experiment, not a metric reconstruction guarantee.
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from artifacts.research.experiments.mapanything_mac import digest, processed_sources
from lingbot_map.reconstruction.fusion import fuse
from lingbot_map.reconstruction.validation import validate


def pool(tensor):
    return F.avg_pool2d(tensor, 5, stride=1, padding=2, count_include_pad=False)


def neighbors_for(frame_ids, index):
    return [
        j
        for j in range(max(0, index - 3), min(len(frame_ids), index + 4))
        if j != index and int(frame_ids[j]) % 10 != 5
    ]


def refine_frame(data, index, device="mps", hypotheses=65, occlusion=True):
    """Return corrected depth and diagnostics, without changing inputs."""
    frame = int(data["frame_ids"][index])
    neighbors = neighbors_for(data["frame_ids"], index)
    prior_np = data["depth"][index]
    report = {
        "frame": frame,
        "source_frames": [int(data["frame_ids"][j]) for j in neighbors],
    }
    if frame % 10 == 5 or len(neighbors) < 2:
        return prior_np.copy(), {
            **report,
            "changed_fraction": 0,
            "skipped": "reserved or fewer than two sources",
        }
    started = time.monotonic()
    dtype = torch.float32
    prior = torch.as_tensor(prior_np, dtype=dtype, device=device)
    h, w = prior.shape
    images = (
        torch.as_tensor(data["rgb"], dtype=dtype, device=device).permute(0, 3, 1, 2)
        / 255
    )
    target = images[index : index + 1]
    gray_weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
    target_gray = (target * gray_weights).sum(1, keepdim=True)
    target_mean = pool(target_gray)
    target_variance = (pool(target_gray.square()) - target_mean.square()).clamp_min(0)
    y, x = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    pixels = torch.stack([x, y, torch.ones_like(x)], -1).to(dtype)
    inverse_k = torch.as_tensor(
        np.linalg.inv(data["intrinsics"][index]), device=device, dtype=dtype
    )
    rays = pixels @ inverse_k.T
    reference_e = np.eye(4)
    reference_e[:3] = data["extrinsics"][index]
    source_info = []
    for j in neighbors:
        source_e = np.eye(4)
        source_e[:3] = data["extrinsics"][j]
        relative = source_e @ np.linalg.inv(reference_e)
        source_info.append(
            (
                j,
                torch.as_tensor(relative[:3, :3], dtype=dtype, device=device),
                torch.as_tensor(relative[:3, 3], dtype=dtype, device=device),
                torch.as_tensor(data["intrinsics"][j], dtype=dtype, device=device),
                torch.as_tensor(data["depth"][j], dtype=dtype, device=device)[
                    None, None
                ],
                torch.as_tensor(
                    data["confidence"][j] >= np.quantile(data["confidence"][j], 0.75),
                    device=device,
                    dtype=dtype,
                )[None, None],
            )
        )
    factors = torch.exp(
        torch.linspace(-math.log(2), math.log(2), hypotheses, device=device)
    )
    costs = []
    observable_masks = []
    for _, rotation, translation, k, _, _ in source_info:
        endpoints = (
            rays[None]
            * prior[None, ..., None]
            * torch.tensor([0.5, 2.0], device=device)[:, None, None, None]
        )
        camera = endpoints @ rotation.T + translation
        projected = camera @ k.T
        uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-8)
        observable_masks.append(
            (torch.linalg.vector_norm(uv[1] - uv[0], dim=-1) >= 1)
            & (camera[..., 2] > 0).all(0)
        )
    for start in range(0, hypotheses, 8):
        scales = factors[start : start + 8]
        depth = prior[None] * scales[:, None, None]
        points = rays[None] * depth[..., None]
        source_costs = []
        for source_index, (
            j,
            rotation,
            translation,
            k,
            source_depth,
            confident,
        ) in enumerate(source_info):
            camera = points @ rotation.T + translation
            projected = camera @ k.T
            uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-8)
            grid = uv * torch.tensor([2 / (w - 1), 2 / (h - 1)], device=device) - 1
            valid = (
                (camera[..., 2] > 0)
                & (uv[..., 0] >= 2)
                & (uv[..., 0] < w - 3)
                & (uv[..., 1] >= 2)
                & (uv[..., 1] < h - 3)
            )
            valid &= observable_masks[source_index][None]
            if occlusion:
                observed = F.grid_sample(
                    source_depth.expand(len(scales), -1, -1, -1),
                    grid,
                    align_corners=True,
                )[:, 0]
                high_confidence = (
                    F.grid_sample(
                        confident.expand(len(scales), -1, -1, -1),
                        grid,
                        mode="nearest",
                        align_corners=True,
                    )[:, 0]
                    > 0.5
                )
                valid &= ~(
                    high_confidence
                    & (observed > 0)
                    & (camera[..., 2] > observed * 1.08)
                )
            warped = F.grid_sample(
                images[j : j + 1].expand(len(scales), -1, -1, -1),
                grid,
                align_corners=True,
            )
            gray = (warped * gray_weights).sum(1, keepdim=True)
            mean = pool(gray)
            variance = (pool(gray.square()) - mean.square()).clamp_min(0)
            covariance = pool(gray * target_gray) - mean * target_mean
            correlation = (
                covariance / (variance * target_variance).clamp_min(1e-10).sqrt()
            )
            cost = 0.8 * (1 - correlation.clamp(-1, 1)) / 2 + 0.2 * pool(
                (warped - target).abs().mean(1, keepdim=True)
            )
            patch_valid = pool(valid[:, None].to(dtype)) >= 0.99999
            source_costs.append(torch.where(patch_valid, cost, 1000)[:, 0])
        # Best two sources tolerate occlusion in others, but never a single-view match.
        best_two = torch.topk(torch.stack(source_costs), 2, dim=0, largest=False).values
        costs.append(best_two.mean(0))
    volume = torch.cat(costs)
    best_cost, best_index = volume.min(0)
    best_factor = factors[best_index]
    baseline_cost = volume[hypotheses // 2]
    separated = torch.abs(
        torch.log(factors[:, None, None] / best_factor[None])
    ) > math.log(1.06)
    runner_up = torch.where(separated, volume, 1000).min(0).values
    margin = runner_up - best_cost
    accepted = (
        (target_variance[0, 0] > 1e-4)
        & (best_cost < 0.25)
        & (baseline_cost - best_cost > 0.03)
        & (margin > 0.01)
    )
    accepted &= torch.isfinite(prior) & (prior > 0)
    accepted[:2] = False
    accepted[-2:] = False
    accepted[:, :2] = False
    accepted[:, -2:] = False
    result = torch.where(accepted, prior * best_factor, prior).cpu().numpy()
    changed = accepted.cpu().numpy()
    baseline = baseline_cost.cpu().numpy()
    best = best_cost.cpu().numpy()
    factors_np = best_factor.cpu().numpy()
    valid = baseline < 1
    report.update(
        {
            "changed_fraction": float(changed.mean()),
            "observable_fraction": float(
                (torch.stack(observable_masks).sum(0) >= 2).float().mean().cpu()
            ),
            "accepted_from_occluded_prior_fraction": float(
                (changed & (baseline >= 1)).mean()
            ),
            "valid_photometric_fraction": float(valid.mean()),
            "baseline_cost_median_valid": float(np.median(baseline[valid]))
            if valid.any()
            else None,
            "best_cost_median_valid": float(np.median(best[valid]))
            if valid.any()
            else None,
            "accepted_depth_factor_p5_p50_p95": np.quantile(
                factors_np[changed], [0.05, 0.5, 0.95]
            ).tolist()
            if changed.any()
            else [],
            "seconds": time.monotonic() - started,
        }
    )
    return result.astype(np.float32), report


def publish(root, data, manifest, inference):
    root.mkdir(parents=True, exist_ok=True)
    (root / "windows").mkdir(exist_ok=True)
    archive = root / "windows/000000.npz"
    if archive.exists():
        with np.load(archive) as saved:
            if any(
                not np.array_equal(saved[key], value) for key, value in data.items()
            ):
                raise ValueError(
                    "Existing prediction archive changed; preserve it and use a new output"
                )
    else:
        with archive.open("xb") as stream:
            np.savez_compressed(stream, **data)
    for filename, value in [("input.json", manifest), ("inference.json", inference)]:
        path = root / filename
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError(f"Existing {filename} has different inputs")
        else:
            with path.open("x") as stream:
                json.dump(value, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=760)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--hypotheses", type=int, default=65)
    parser.add_argument("--no-occlusion", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.start % 10 or args.count < 8:
        parser.error(
            "Require nonnegative start divisible by 10 and at least eight frames"
        )
    if args.hypotheses < 9 or args.hypotheses % 2 != 1:
        parser.error(
            "Require an odd hypothesis count >= 9, preserving the unchanged hypothesis"
        )
    manifest = json.loads((args.source / "input.json").read_text())
    selected = manifest["frames"][args.start : args.start + args.count]
    if len(selected) != args.count:
        parser.error("Requested frames are unavailable")
    paths, hashes = processed_sources(args.source, selected)
    signature = {
        "schema": 1,
        "algorithm_sha256": digest(__file__),
        "input_sha256": digest(args.source / "input.json"),
        "camera_assignment_sha256": digest(args.source / "model/cameras.json"),
        "camera_windows": hashes,
        "start": args.start,
        "count": args.count,
        "device": args.device,
        "hypotheses": args.hypotheses,
        "occlusion": not args.no_occlusion,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    signature_path = args.output / "signature.json"
    if signature_path.exists():
        if json.loads(signature_path.read_text()) != signature:
            raise ValueError("Output belongs to a different experiment")
    else:
        with signature_path.open("x") as stream:
            json.dump(signature, stream, indent=2)
    arrays = {
        key: [] for key in ["depth", "confidence", "extrinsics", "intrinsics", "rgb"]
    }
    previous, window = None, None
    for path, frame in zip(paths, selected):
        if path != previous:
            with np.load(path) as archive:
                window = dict(archive)
            previous = path
        index = int(np.flatnonzero(window["frame_ids"] == frame["id"])[0])
        for key, values in arrays.items():
            values.append(window[key][index])
    data = {key: np.stack(value) for key, value in arrays.items()}
    data["frame_ids"] = np.arange(args.count)
    manifest = copy.deepcopy(manifest)
    manifest["frames"] = [
        {
            **frame,
            "id": i,
            "original_frame": frame["id"],
            "file": str((args.source / frame["file"]).resolve()),
        }
        for i, frame in enumerate(selected)
    ]
    manifest["configuration"]["temporal_subset"] = {
        "start_frame": args.start,
        "stop_frame": args.start + args.count,
    }
    inference = {
        "window": args.count,
        "overlap": min(24, args.count // 2),
        "poses_global": True,
        "pose_source": "Fixed Cauchy bundle cameras",
        "photometric_experiment": signature,
    }
    baseline = args.output / "baseline"
    publish(baseline, data, manifest, inference)
    results = {}
    for arm in ["baseline", "refined"]:
        root = args.output / arm
        if arm == "refined":
            cache = args.output / "depth"
            cache.mkdir(exist_ok=True)
            corrected = data["depth"].copy()
            with (args.output / "trace.jsonl").open("a") as trace:
                for i in range(args.count):
                    if i % 10 == 5:
                        continue
                    path = cache / f"{args.start + i:06d}.npz"
                    if path.exists():
                        with np.load(path) as saved:
                            corrected[i] = saved["depth"]
                        continue
                    with torch.inference_mode():
                        corrected[i], report = refine_frame(
                            data, i, args.device, args.hypotheses, not args.no_occlusion
                        )
                    report["original_frame"] = args.start + i
                    with path.open("xb") as stream:
                        np.savez_compressed(
                            stream, depth=corrected[i], report=json.dumps(report)
                        )
                    trace.write(json.dumps(report) + "\n")
                    trace.flush()
                    print(json.dumps(report), flush=True)
                    if args.device == "mps":
                        torch.mps.empty_cache()
            refined = {**data, "depth": corrected}
            publish(
                root,
                refined,
                manifest,
                {**inference, "photometric_depth_corrected": True},
            )
        fuse(root)
        results[arm] = {}
        for label, asset in [("full", None), ("app", "property.glb")]:
            report = validate(root, asset_name=asset)
            results[arm][label] = {k: v for k, v in report.items() if k != "views"}
        with (args.output / f"{arm}-results.json").open("w") as stream:
            json.dump(results[arm], stream, indent=2)
        print(json.dumps({"arm": arm, "results": results[arm]}), flush=True)


if __name__ == "__main__":
    main()
