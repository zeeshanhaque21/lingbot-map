"""Run the released FixAnything model with explicit local MPS adaptations."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from lingbot_map.reconstruction.fixanything_mps import install_mps_compatibility
from lingbot_map.reconstruction.io import digest, write_json


def inspect_sweep(sweep, num_frames):
    """Reject missing camera-control renders before loading model weights.

    Passing this input check does not establish generated panorama quality.
    """
    sweep = Path(sweep)
    config = json.loads((sweep / "sweep.json").read_text())
    frames = config["frames"]
    clean = config["clean_frame_indices"]
    errors = []
    if num_frames != 61 or len(frames) != 61:
        errors.append("A pilot must use the complete 61-view sweep")
    if any(not isinstance(i, int) or i < 0 or i >= len(frames) for i in clean):
        errors.append("Clean anchor indices fall outside the sweep")
    marked = [i for i, frame in enumerate(frames) if frame["captured_anchor"]]
    if sorted(set(clean)) != marked:
        errors.append("Clean anchor indices disagree with captured-anchor metadata")
    empty = [
        i
        for i, frame in enumerate(frames)
        if i not in clean and frame["observed_fraction"] == 0
    ]
    if empty:
        errors.append(
            f"{len(empty)} rendered views have no observed surfaces and therefore "
            "provide no scene-based camera control; repair conditioning first"
        )
    anchor_hashes = set()
    for i in range(len(frames)):
        path = sweep / "frames" / f"{i:03d}.png"
        if not path.exists():
            errors.append(f"Missing input image: {path}")
            continue
        try:
            with Image.open(path) as image:
                if image.size != (config["width"], config["height"]):
                    errors.append(f"Input image dimensions changed: {path}")
                image.verify()
        except (OSError, ValueError) as error:
            errors.append(f"Unreadable input image {path}: {error}")
        if i in clean:
            anchor_hashes.add(digest(path))
    return {
        "ready_for_pilot": not errors,
        "errors": errors,
        "frames": len(frames),
        "requested_frames": num_frames,
        "clean_frame_indices": clean,
        "unique_clean_images": len(anchor_hashes),
        "sweep_mode": config.get("sweep_mode", "spherical"),
        "coverage_target": config.get("coverage_target", "sphere"),
        "empty_render_indices": empty,
        "under_ten_percent_observed_frames": sum(
            frame["observed_fraction"] < 0.1 for frame in frames
        ),
        "sweep_sha256": digest(sweep / "sweep.json"),
        "panorama_quality_verified": False,
        "interpretation": "Input checks only. Repeated anchors do not add new views. A complete single-station generation, stitch and source-photo review must pass before processing further stations.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check-input",
        action="store_true",
        help="Inspect a complete 61-view pilot without loading model weights",
    )
    parser.add_argument(
        "--repository", type=Path, default=Path.home() / "Projects/fix-anything"
    )
    parser.add_argument("--models", type=Path, default=Path("checkpoints/fixanything"))
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=61)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    inspection = inspect_sweep(args.sweep, args.frames)
    if args.check_input:
        print(json.dumps(inspection, indent=2))
        return 0 if inspection["ready_for_pilot"] else 2
    if not inspection["ready_for_pilot"]:
        parser.error("; ".join(inspection["errors"]))
    if args.output is None:
        parser.error("--output is required for generation")
    if args.output.exists():
        parser.error("Preserve existing inference outputs; choose a new directory")
    if args.frames < 1 or args.frames > 61 or args.frames % 4 != 1 or args.steps < 1:
        parser.error(
            "Use 1 to 61 frames, with frame count congruent to 1 modulo 4, and positive steps"
        )
    if not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    if min(args.width, args.height) < 64 or args.width % 16 or args.height % 16:
        parser.error("Image dimensions must be multiples of 16 and at least 64")
    manifest = json.loads((args.models / "download-manifest.json").read_text())
    incomplete = [
        item["path"]
        for item in manifest
        if not Path(item["path"]).exists()
        or Path(item["path"]).stat().st_size != item["bytes"]
        or Path(item["path"] + ".aria2").exists()
    ]
    if incomplete:
        parser.error(f"Model downloads are incomplete: {incomplete}")
    if not (args.models / "download-completion.json").exists():
        parser.error(
            "Run fetch_fixanything_models.py --status to verify Motrix completion"
        )
    args.output.mkdir(parents=True)
    started = time.time()
    source = args.repository / "scripts/run_inference.py"
    spec = importlib.util.spec_from_file_location("released_fixanything", source)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    changes = install_mps_compatibility()
    config = {
        "pid": os.getpid(),
        "started_unix": started,
        "torch": torch.__version__,
        "device": "mps",
        "dtype": "bfloat16",
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "adaptations": changes,
        "fixanything_commit": subprocess.check_output(
            ["git", "-C", str(args.repository), "rev-parse", "HEAD"], text=True
        ).strip(),
        "script_sha256": digest(source),
        "model_manifest_sha256": digest(args.models / "download-manifest.json"),
        "download_receipt_sha256": digest(args.models / "download-completion.json"),
        "generated": True,
        "input_inspection": inspection,
    }
    write_json(args.output / "started.json", config)
    log = args.output / "events.jsonl"

    def event(stage, **extra):
        record = {"stage": stage, "elapsed_seconds": time.time() - started, **extra}
        with log.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)

    try:
        event("loading_models")
        pipe = api.WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="mps",
            model_configs=[
                api._model_config(str(args.models), p, offload_device="cpu")
                for p in api.WAN_MODEL_FILES
            ],
            tokenizer_config=api._model_config(
                str(args.models), api.WAN_TOKENIZER_FILES
            ),
            redirect_common_files=False,
        )
        event("loading_adapter")
        pipe.load_lora(
            pipe.dit, str(args.models / "fixanything_lora.safetensors"), alpha=1.0
        )
        # Fixed budget bypasses CUDA-only dynamic memory discovery on this 128 GB Mac.
        pipe.enable_vram_management(num_persistent_param_in_dit=100_000_000_000)
        scheduler_step = pipe.scheduler.step
        step_index = 0

        def saved_step(*step_args, **step_kwargs):
            nonlocal step_index
            result = scheduler_step(*step_args, **step_kwargs)
            # Denoising can be long; preserve each completed latent state.
            torch.save(
                result.detach().cpu(), args.output / f"latents-{step_index:03d}.pt"
            )
            event(
                "denoising_step",
                index=step_index,
                mps_driver_bytes=torch.mps.driver_allocated_memory(),
            )
            step_index += 1
            return result

        pipe.scheduler.step = saved_step
        paths = sorted((args.sweep / "frames").glob("*.png"))
        frames = [Image.open(path).convert("RGB") for path in paths[: args.frames]]
        clean = inspection["clean_frame_indices"]
        event(
            "inference",
            frames=args.frames,
            width=args.width,
            height=args.height,
            steps=args.steps,
        )
        generated, inputs = api.fix_video(
            pipe,
            frames,
            clean_frame_indices=clean,
            num_frames=args.frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            seed=args.seed,
        )
        (args.output / "frames").mkdir()
        for index, frame in enumerate(generated):
            frame.save(args.output / "frames" / f"{index:03d}.png")
            event("saved_frame", index=index)
        api.save_video(generated, str(args.output / "generated.mp4"), fps=15)
        api.save_video(
            api.side_by_side(inputs, generated),
            str(args.output / "comparison.mp4"),
            fps=15,
        )
        event("complete")
        write_json(
            args.output / "completion.json",
            {
                "success": True,
                "seconds": time.time() - started,
                "frames": len(generated),
                "generated": True,
                "complete_sweep": len(generated) == args.frames,
                "sweep_mode": inspection["sweep_mode"],
                "coverage_target": inspection["coverage_target"],
                "full_spherical_sweep": len(generated) == 61
                and inspection["coverage_target"] == "sphere",
            },
        )
    except BaseException as error:
        event("failed", error=repr(error))
        write_json(
            args.output / "completion.json",
            {"success": False, "seconds": time.time() - started, "error": repr(error)},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
