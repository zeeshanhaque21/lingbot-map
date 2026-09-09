"""Jointly refine cameras, shared landmarks and regularized depth scales.

Fits image reprojections directly instead of averaging independently fitted
rigid transforms. The first camera and first depth scale fix the coordinate
gauge. One fifth of tracks remain reserved from all fitting.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lingbot_map.reconstruction.io import digest, write_json, write_npz

from .joint_pose_graph import evaluate, publish


class BundleProblem(torch.nn.Module):
    def __init__(self, data, cameras, device="cpu"):
        super().__init__()
        self.initial = np.asarray([c["camera_to_world"] for c in cameras])
        frame_index = {c["frame"]: i for i, c in enumerate(cameras)}
        ids, counts = np.unique(data["track"], return_counts=True)
        ids = ids[(ids % 5 != 0) & (counts >= 3)]
        selected = np.isin(data["track"], ids)
        if len(ids) < 12:
            raise ValueError(
                "Need at least twelve training landmarks seen in three views"
            )
        self.unit = float(np.median(data["xyz"][selected, 2]))
        point_index = np.searchsorted(ids, data["track"][selected])
        frames = np.asarray([frame_index[int(f)] for f in data["frame"][selected]])
        world = (
            np.einsum("nij,nj->ni", self.initial[frames, :3, :3], data["xyz"][selected])
            + self.initial[frames, :3, 3]
        ) / self.unit
        points = np.zeros((len(ids), 3))
        np.add.at(points, point_index, world)
        points /= np.bincount(point_index)[:, None]
        k = np.asarray([c["intrinsics"] for c in cameras])
        for name, value, dtype in [
            ("frame_index", frames, torch.long),
            ("point_index", point_index, torch.long),
            ("target_pixels", data["pixel"][selected], torch.float32),
            ("reference_depth", data["xyz"][selected, 2] / self.unit, torch.float32),
            ("base_rotation", self.initial[:, :3, :3], torch.float32),
            ("base_center", self.initial[:, :3, 3] / self.unit, torch.float32),
            ("focal", k[:, [0, 1], [0, 1]], torch.float32),
            ("principal", k[:, :2, 2], torch.float32),
        ]:
            self.register_buffer(
                name, torch.as_tensor(value, dtype=dtype, device=device)
            )
        self.rotation_delta = torch.nn.Parameter(
            torch.zeros((len(cameras) - 1, 3), device=device)
        )
        self.center_delta = torch.nn.Parameter(
            torch.zeros((len(cameras) - 1, 3), device=device)
        )
        self.landmarks = torch.nn.Parameter(
            torch.as_tensor(points, dtype=torch.float32, device=device)
        )
        self.log_scale = torch.nn.Parameter(
            torch.zeros(len(cameras) - 1, device=device)
        )

    def camera_state(self):
        zero = torch.zeros((1, 3), device=self.rotation_delta.device)
        vector = torch.cat([zero, self.rotation_delta], dim=0)
        q = torch.cat([torch.ones_like(vector[:, :1]), vector], dim=1)
        q = q / torch.linalg.vector_norm(q, dim=1, keepdim=True)
        w, x, y, z = q.unbind(1)
        delta = torch.stack(
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
            dim=1,
        ).reshape(-1, 3, 3)
        rotation = self.base_rotation @ delta
        center = self.base_center + torch.cat([zero, self.center_delta], dim=0)
        scales = torch.cat([self.log_scale.new_zeros(1), self.log_scale])
        return rotation, center, scales

    def loss_terms(self):
        rotation, center, scales = self.camera_state()
        offsets = self.landmarks[self.point_index] - center[self.frame_index]
        local = torch.bmm(offsets[:, None, :], rotation[self.frame_index]).squeeze(1)
        depth = local[:, 2].clamp_min(1e-4)
        pixels = (
            local[:, :2] / depth[:, None] * self.focal[self.frame_index]
            + self.principal[self.frame_index]
        )
        residual = pixels - self.target_pixels
        pixel_loss = (torch.sqrt(1 + (residual / 2).square().sum(1)) - 1).mean()
        log_error = torch.log(depth / self.reference_depth) - scales[self.frame_index]
        depth_loss = (torch.sqrt(1 + (log_error / 0.05).square()) - 1).mean()
        scale_prior = (scales / 0.05).square().mean()
        scale_smoothness = (torch.diff(scales) / 0.02).square().mean()
        rotation_prior = (2 * self.rotation_delta / 0.05).square().mean()
        center_prior = (self.center_delta / 0.1).square().mean()
        behind = torch.relu(0.01 - local[:, 2]).square().mean()
        loss = (
            pixel_loss
            + 0.2 * depth_loss
            + 0.2 * scale_prior
            + 0.05 * scale_smoothness
            + 0.01 * rotation_prior
            + 0.01 * center_prior
            + 100 * behind
        )
        return loss, residual, log_error

    def export(self):
        with torch.no_grad():
            rotation, center, scales = self.camera_state()
            poses = np.repeat(np.eye(4)[None], len(rotation), axis=0)
            poses[:, :3, :3] = rotation.cpu().numpy()
            poses[:, :3, 3] = center.cpu().numpy() * self.unit
            return poses, scales.exp().cpu().numpy()


def fit_bundle(problem, output, steps):
    state_path = output / "bundle-state.npz"
    start = 0
    if state_path.exists():
        saved = dict(np.load(state_path))
        with torch.no_grad():
            for name, parameter in problem.named_parameters():
                parameter.copy_(torch.as_tensor(saved[name], device=parameter.device))
        start = int(saved["completed_steps"])
    optimizer = torch.optim.LBFGS(
        problem.parameters(),
        max_iter=20,
        max_eval=30,
        history_size=30,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )
    started = time.monotonic()
    for step in range(start, steps):

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = problem.loss_terms()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite bundle objective")
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            loss, pixels, log_error = problem.loss_terms()
            error = torch.linalg.vector_norm(pixels, dim=1).cpu().numpy()
            record = {
                "step": step + 1,
                "loss": float(loss),
                "training_pixel_error_p50_p90_p95": np.quantile(
                    error, [0.5, 0.9, 0.95]
                ).tolist(),
                "median_absolute_log_depth_error": float(torch.median(log_error.abs())),
                "seconds": time.monotonic() - started,
                "resumed_without_optimizer_history": bool(start),
            }
            arrays = {
                name: parameter.detach().cpu().numpy()
                for name, parameter in problem.named_parameters()
            }
            write_npz(state_path, **arrays, completed_steps=np.array(step + 1))
        with (output / "trace.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
        print(
            f"bundle_step: {step + 1}\nloss: {record['loss']:.5f}\ntraining_pixel_median: {record['training_pixel_error_p50_p90_p95'][0]:.3f}",
            flush=True,
        )
        if problem.rotation_delta.device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
    poses, scales = problem.export()
    if not np.isfinite(poses).all() or not np.isfinite(scales).all():
        raise ValueError("Bundle refinement produced non-finite geometry")
    write_npz(output / "optimized-geometry.npz", poses=poses, scales=scales)
    return poses, scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="Joint pose graph observation directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; use --device cpu")
    cached = json.loads((args.cache / "input.json").read_text())
    source_signature = {
        "source_inference": digest(args.source / "inference.json"),
        "source_cameras": digest(args.source / "model/cameras.json"),
        "source_windows": {
            p.name: digest(p) for p in sorted((args.source / "windows").glob("*.npz"))
        },
    }
    if any(cached.get(key) != value for key, value in source_signature.items()):
        parser.error("--cache belongs to different source cameras or depths")
    signature = {
        "version": 1,
        "method": "direct_reprojection_bundle",
        "steps": args.steps,
        "device": args.device,
        "source": source_signature,
        "observations_sha256": digest(args.cache / "observations.npz"),
        "pixel_scale": 2.0,
        "depth_sigma": 0.05,
        "depth_weight": 0.2,
        "scale_sigma": 0.05,
        "scale_weight": 0.2,
        "scale_smooth_sigma": 0.02,
        "scale_smooth_weight": 0.05,
        "rotation_sigma": 0.05,
        "rotation_weight": 0.01,
        "center_sigma": 0.1,
        "center_weight": 0.01,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "input.json"
    if marker.exists() and json.loads(marker.read_text()) != signature:
        parser.error("Inputs changed; choose a new --output")
    write_json(marker, signature)
    data = dict(np.load(args.cache / "observations.npz"))
    cameras = json.loads((args.source / "model/cameras.json").read_text())
    problem = BundleProblem(data, cameras, args.device)
    print(
        f"training_landmarks: {len(problem.landmarks)}\ntraining_observations: {len(problem.frame_index)}",
        flush=True,
    )
    poses, scales = fit_bundle(problem, args.output, args.steps)
    del problem
    if args.device == "mps":
        torch.mps.empty_cache()
    initial = np.asarray([c["camera_to_world"] for c in cameras])
    scaled = dict(data)
    scale_by_frame = {c["frame"]: scale for c, scale in zip(cameras, scales)}
    scaled["xyz"] = (
        data["xyz"] * np.array([scale_by_frame[int(f)] for f in data["frame"]])[:, None]
    )
    write_json(
        args.output / "camera-comparison.json",
        {
            "baseline": evaluate(data, cameras, initial),
            "optimized": evaluate(scaled, cameras, poses),
            "depth_scale_p0_p5_p50_p95_p100": np.quantile(
                scales, [0, 0.05, 0.5, 0.95, 1]
            ).tolist(),
            "metric_accuracy_verified": False,
        },
    )
    candidate = publish(
        args.source,
        args.output,
        cameras,
        poses,
        signature,
        scales,
        pose_source="Direct image bundle adjustment with regularized depth scales",
    )
    from lingbot_map.reconstruction.fusion import fuse
    from lingbot_map.reconstruction.validation import validate

    fuse(candidate)
    results = {}
    for name, asset in [("full", None), ("app", "property.glb")]:
        report = validate(candidate, asset_name=asset)
        results[name] = {k: v for k, v in report.items() if k != "views"}
        results[name]["failed_views"] = [
            v["frame"] for v in report["views"] if v["supported_depth_fraction"] < 0.4
        ]
        write_json(args.output / "results.json", results)
        print(
            f"asset: {name}\nview_consistency_gate: {str(report['view_consistency_gate']).lower()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
