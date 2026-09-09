"""Bounded native PyTorch inference with independently resumable windows."""
import contextlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from .io import digest, write_json, write_npz


def window_ranges(count, size, overlap):
    if not 0 < overlap < size or size < 8:
        raise ValueError("Require window >= 8 and 0 < overlap < window")
    start = 0
    while start < count:
        end = min(count, start + size)
        yield start, end
        if end == count:
            break
        start = end - overlap


def infer(output, checkpoint, window=96, overlap=24, device="mps", precision="float32"):
    output = Path(output)
    manifest = json.loads((output / "input.json").read_text())
    frames = manifest["frames"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable; use --device cpu or an Apple Silicon Python installation")
    configuration = {"checkpoint_sha256": digest(checkpoint), "window": window,
                     "overlap": overlap, "device": device, "precision": precision,
                     "image_size": 518, "scale_frames": 8, "cache_frames": 64,
                     "input_sha256": digest(output / "input.json"), "schema": 1}
    config_path = output / "inference.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Inference configuration changed; choose a new --output")
    if not config_path.exists():
        write_json(config_path, configuration)
    model = None
    for start, end in window_ranges(len(frames), window, overlap):
        destination = output / "windows" / f"{start:06d}.npz"
        if destination.exists():
            with np.load(destination) as saved:
                if saved["frame_ids"].tolist() != list(range(start, end)):
                    raise ValueError(f"Invalid saved window: {destination}")
            continue
        if model is None:
            print("Loading model on " + device, file=sys.stderr, flush=True)
            model = GCTStream(img_size=518, patch_size=14, enable_3d_rope=True,
                              max_frame_num=max(1024, window), kv_cache_sliding_window=64,
                              kv_cache_scale_frames=8, use_sdpa=True).eval()
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = state.get("model", state)
            model.load_state_dict(state, strict=True)
            del state
            model.to(device)
            if device == "mps":
                # Dynamic attention shapes can leave a large pool of unused Metal
                # allocations. Release only unused buffers, preserving the live KV cache.
                original_forward = model.forward
                trace = output / "memory.jsonl"
                def forward_with_memory_release(*args, **kwargs):
                    result = original_forward(*args, **kwargs)
                    torch.mps.synchronize()
                    allocated = torch.mps.current_allocated_memory()
                    driver_before = torch.mps.driver_allocated_memory()
                    torch.mps.empty_cache()
                    with trace.open("a") as stream:
                        stream.write(json.dumps({"allocated_bytes": allocated,
                            "driver_before_release": driver_before,
                            "driver_after_release": torch.mps.driver_allocated_memory()}) + "\n")
                        stream.flush()
                    return result
                model.forward = forward_with_memory_release
        images = load_and_preprocess_images(
            [str(output / frame["file"]) for frame in frames[start:end]],
            image_size=518, patch_size=14)
        began = time.monotonic()
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(precision)
        amp = torch.autocast(device_type=device, dtype=dtype) if dtype else contextlib.nullcontext()
        with torch.inference_mode(), amp:
            prediction = model.inference_streaming(images, num_scale_frames=min(8, end-start),
                                                   output_device=torch.device("cpu"))
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            prediction["pose_enc"].float(), image_size_hw=images.shape[-2:])
        depth = prediction["depth"][0, ..., 0].float().numpy()
        confidence = prediction["depth_conf"][0].float().numpy()
        if not np.isfinite(depth).all() or not np.isfinite(extrinsics.numpy()).all():
            raise ValueError(f"Non-finite model output at frame {start}; try --precision float32")
        write_npz(destination, frame_ids=np.arange(start, end), depth=depth, confidence=confidence,
                  extrinsics=extrinsics[0].numpy(), intrinsics=intrinsics[0].numpy(),
                  rgb=(images.permute(0, 2, 3, 1).numpy()*255).round().clip(0,255).astype(np.uint8))
        write_json(destination.with_suffix(".json"), {
            "start": start, "end": end, "seconds": time.monotonic()-began,
            "sha256": digest(destination), "state": "complete"})
        del prediction, images
        if device == "mps":
            torch.mps.empty_cache()
        print(f"Saved frames {start}:{end}", file=sys.stderr, flush=True)
