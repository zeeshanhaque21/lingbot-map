"""Run local MapAnything weights on captured frames, preserving provenance.

Experimental adapter, not a claim of metric or building accuracy.
Use the separate .venv-mapanything environment for inference and the existing
.venv-reconstruction environment for fusion and validation.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
import torch


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


@contextmanager
def local_dino_hub(repository):
    """Only the local DINO architecture is allowed; full weights are already local."""
    original = torch.hub.load

    def load(repo_or_dir, model, *args, **kwargs):
        if (
            repo_or_dir != "facebookresearch/dinov2"
            or kwargs.get("pretrained") is not False
        ):
            raise ValueError(
                "Unexpected torch.hub request; no automatic downloads allowed"
            )
        kwargs.pop("force_reload", None)
        return original(str(repository), model, *args, source="local", **kwargs)

    with patch.object(torch.hub, "load", load):
        yield


def pack_predictions(predictions):
    def array(key):
        return np.stack([p[key][0].detach().float().cpu().numpy() for p in predictions])

    depth = array("depth_z")[..., 0]
    poses = array("camera_poses")
    intrinsics = array("intrinsics")
    confidence = array("conf")
    mask = array("mask")[..., 0].astype(bool)
    rgb = (array("img_no_norm") * 255).round().clip(0, 255).astype(np.uint8)
    if not all(np.isfinite(x).all() for x in (depth, poses, intrinsics, confidence)):
        raise ValueError("Nonfinite predictions; preserve run for diagnosis")
    rotation = poses[:, :3, :3]
    if not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("Predicted rotations are not orthonormal")
    if not np.allclose(np.linalg.det(rotation), 1, atol=1e-4):
        raise ValueError("Predicted poses contain reflected or scaled rotations")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Invalid homogeneous camera poses")
    # Report how closely the pinhole export matches the model's native ray field.
    h, w = depth.shape[1:]
    y, x = np.indices((h, w))
    pixels = np.stack([x, y, np.ones_like(x)], -1)
    camera = (
        np.einsum("hwj,nij->nhwi", pixels, np.linalg.inv(intrinsics)) * depth[..., None]
    )
    native_camera = array("pts3d_cam")
    positive = (depth > 0) & mask
    if not positive.any():
        raise ValueError("No positive unmasked depth predictions")
    relative = np.linalg.norm(camera - native_camera, axis=-1) / np.maximum(depth, 1e-8)
    native_world = array("pts3d")
    reconstructed_world = (
        np.einsum("nhwj,nij->nhwi", native_camera, rotation)
        + poses[:, None, None, :3, 3]
    )
    world_error = np.linalg.norm(reconstructed_world - native_world, axis=-1)
    report = {
        "pinhole_relative_error_p50_p95_p99": np.quantile(
            relative[positive], [0.5, 0.95, 0.99]
        ).tolist(),
        "native_c2w_world_error_max": float(world_error.max()),
        "native_c2w_world_error_relative_p99": float(
            np.quantile(world_error[positive] / depth[positive], 0.99)
        ),
        "mask_fraction": float(mask.mean()),
        "depth_p5_p50_p95": np.quantile(depth[positive], [0.05, 0.5, 0.95]).tolist(),
    }
    if report["native_c2w_world_error_relative_p99"] > 1e-4:
        raise ValueError(f"C2W normalization disagrees with native points: {report}")
    return {
        "frame_ids": np.arange(len(predictions)),
        "depth": depth,
        "confidence": np.where(mask, confidence, -1).astype(np.float32),
        "extrinsics": np.linalg.inv(poses)[:, :3].astype(np.float32),
        "intrinsics": intrinsics,
        "rgb": rgb,
        "prediction_mask": mask,
    }, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dino-repository", type=Path, required=True)
    parser.add_argument("--start", type=int, default=760)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    source = args.source.resolve()
    manifest = json.loads((source / "input.json").read_text())
    selected = manifest["frames"][args.start : args.start + args.count]
    if args.start < 0 or args.count < 2 or len(selected) != args.count:
        parser.error("Requested frame range must contain at least two available frames")
    if not (args.dino_repository / "hubconf.py").is_file():
        parser.error("A local DINOv2 checkout is required")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    paths = [(source / frame["file"]).resolve() for frame in selected]
    signature = {
        "model": "facebook/map-anything-apache",
        "weights_directory": str(args.weights.resolve()),
        "checkpoint_sha256": digest(args.weights / "model.safetensors"),
        "config_sha256": digest(args.weights / "config.json"),
        "input_sha256": digest(source / "input.json"),
        "frame_sha256": [digest(path) for path in paths],
        "dino_commit": subprocess.check_output(
            ["git", "-C", str(args.dino_repository), "rev-parse", "HEAD"], text=True
        ).strip(),
        "start": args.start,
        "count": args.count,
        "device": args.device,
        "precision": "float32",
        "resolution": [518, 294],
        "input_modalities": ["images"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    signature_path = args.output / "run-signature.json"
    if signature_path.exists():
        if json.loads(signature_path.read_text()) != signature:
            raise ValueError(
                "Output belongs to another experiment; use a new directory"
            )
        if (args.output / "completed.json").exists():
            saved = json.loads((args.output / "completed.json").read_text())
            if digest(args.output / "windows/000000.npz") != saved["archive_sha256"]:
                raise ValueError("Saved prediction archive changed")
            print(json.dumps(saved), flush=True)
            return
        if (args.output / "windows/000000.npz").exists():
            raise ValueError(
                "Unfinished publication exists; preserve it for inspection"
            )
    else:
        write_new_json(signature_path, signature)

    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    started = time.monotonic()
    print("Loading complete local checkpoint", flush=True)
    with local_dino_hub(args.dino_repository.resolve()):
        model = MapAnything.from_pretrained(
            str(args.weights.resolve()), local_files_only=True, strict=True
        )
    model = model.to(args.device).eval()
    print(f"Loaded model in {time.monotonic() - started:.1f}s", flush=True)
    views = load_images(
        [str(path) for path in paths], resize_mode="fixed_size", size=(518, 294)
    )
    inference_start = time.monotonic()
    with torch.inference_mode():
        predictions = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=False,
            apply_mask=False,
        )
    if args.device == "mps":
        torch.mps.synchronize()
    inference_seconds = time.monotonic() - inference_start
    print(f"Inference completed in {inference_seconds:.1f}s", flush=True)
    arrays, report = pack_predictions(predictions)
    report.update(
        {
            "inference_seconds": inference_seconds,
            "frames": len(predictions),
            "metric_accuracy_verified": False,
        }
    )
    if args.device == "mps":
        report.update(
            {
                "mps_current_bytes": torch.mps.current_allocated_memory(),
                "mps_driver_bytes": torch.mps.driver_allocated_memory(),
            }
        )
    (args.output / "windows").mkdir(exist_ok=True)
    with (args.output / "windows/000000.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    manifest["frames"] = [
        {**frame, "id": i, "original_frame": frame["id"], "file": str(paths[i])}
        for i, frame in enumerate(selected)
    ]
    manifest["configuration"]["temporal_subset"] = {
        "start_frame": args.start,
        "stop_frame": args.start + args.count,
    }
    write_new_json(args.output / "input.json", manifest)
    write_new_json(
        args.output / "inference.json",
        {
            **signature,
            "window": args.count,
            "overlap": 0,
            "image_size": 518,
            "schema": 2,
            "checkpoint_pose_convention": "camera-to-world",
            "stored_pose_convention": "world-to-camera",
            "pose_source": "MapAnything Apache image-only predictions",
        },
    )
    report["archive_sha256"] = digest(args.output / "windows/000000.npz")
    write_new_json(args.output / "completed.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
