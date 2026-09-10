"""Run local MapAnything weights on captured frames, preserving provenance.

Experimental adapter, not a claim of metric or building accuracy.
Use the separate .venv-mapanything environment for inference and the existing
.venv-reconstruction environment for fusion and validation.
"""

import argparse
import hashlib
import json
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
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
    # apply_mask=False retains raw geometry and returns this original mask.
    # The reconstruction pipeline performs its own edge/support filtering.
    mask = array("non_ambiguous_mask").astype(bool)
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


def processed_sources(root, selected):
    cameras = json.loads((root / "model/cameras.json").read_text())
    windows = sorted((root / "windows").glob("*.npz"))
    by_frame = {camera["frame"]: windows[camera["window"]] for camera in cameras}
    paths = [by_frame[frame["id"]] for frame in selected]
    return paths, {str(path.resolve()): digest(path) for path in set(paths)}


def conditioned_views(paths, selected, conditioning):
    """Use the exact owning-window RGB/K/pose triplet, with no resizing ambiguity."""
    views = []
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    data, previous = None, None
    for path, frame in zip(paths, selected):
        if path != previous:
            with np.load(path) as archive:
                data = dict(archive)
            previous = path
        index = int(np.flatnonzero(data["frame_ids"] == frame["id"])[0])
        rgb = torch.from_numpy(data["rgb"][index].copy()).permute(2, 0, 1).float() / 255
        view = {"img": ((rgb - mean) / std)[None], "data_norm_type": ["dinov2"]}
        if conditioning in ("calibration", "posed"):
            view["intrinsics"] = torch.from_numpy(data["intrinsics"][index].copy())[
                None
            ]
        if conditioning == "posed":
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3] = data["extrinsics"][index]
            view["camera_poses"] = torch.from_numpy(np.linalg.inv(w2c))[None]
            view["is_metric_scale"] = False
        views.append(view)
    return views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dino-repository", type=Path, required=True)
    parser.add_argument("--start", type=int, default=760)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument(
        "--processed-source",
        type=Path,
        help="Reuse exact owning-window RGB, calibration and optional poses",
    )
    parser.add_argument(
        "--conditioning", choices=("images", "calibration", "posed"), default="images"
    )
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
    if args.conditioning != "images" and args.processed_source is None:
        parser.error("Conditioning requires --processed-source")
    processed_paths, processed_hashes = [], {}
    if args.processed_source is not None:
        processed_paths, processed_hashes = processed_sources(
            args.processed_source, selected
        )
    signature = {
        "adapter_schema": 2,
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
        "input_modalities": {
            "images": ["images"],
            "calibration": ["images", "intrinsics"],
            "posed": ["images", "intrinsics", "unscaled_poses"],
        }[args.conditioning],
        "processed_window_sha256": processed_hashes,
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
    views = (
        conditioned_views(processed_paths, selected, args.conditioning)
        if processed_paths
        else load_images(
            [str(path) for path in paths], resize_mode="fixed_size", size=(518, 294)
        )
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
            "overlap": min(24, args.count // 2),
            "image_size": 518,
            "schema": 2,
            "checkpoint_pose_convention": "camera-to-world",
            "stored_pose_convention": "world-to-camera",
            "pose_source": f"MapAnything Apache {args.conditioning} predictions",
        },
    )
    report["archive_sha256"] = digest(args.output / "windows/000000.npz")
    write_new_json(args.output / "completed.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
