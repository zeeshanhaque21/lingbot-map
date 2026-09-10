"""Run with python -m lingbot_map.reconstruction."""

import argparse
import contextlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct observed property surfaces on this Mac"
    )
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "infer",
            "run",
            "sfm",
            "densify",
            "normalize",
            "refine",
            "register",
            "repair",
            "export",
            "fuse",
            "validate",
            "view",
            "tour",
            "tour-view",
            "tour-sweep",
            "tour-stitch",
            "tour-attach",
        ],
        nargs="?",
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--video", type=Path)
    inputs.add_argument("--images", type=Path, help="Ordered RGB PNG/JPEG directory; preserves pixels, timing unknown")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--stations", type=int, default=6)
    parser.add_argument("--station", default="000")
    parser.add_argument("--sweep-mode", choices=["spherical", "horizontal"], default="spherical")
    parser.add_argument("--sweep-frames", type=int, default=61, help="Horizontal sweep frame count, including the repeated endpoint")
    parser.add_argument("--sweep-up", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Scene-up direction in reconstruction coordinates; required for horizontal sweeps")
    parser.add_argument("--panorama-width", type=int, default=2048)
    parser.add_argument("--triangle-budget", type=int, default=150000)
    parser.add_argument("--port", type=int)
    parser.add_argument("--generated", action="store_true", help="Mark stitched images as generated")
    parser.add_argument("--colmap-model", type=Path)
    parser.add_argument("--bridges", type=Path)
    parser.add_argument("--camera-source", choices=["sfm", "learned", "native"], default="sfm")
    parser.add_argument("--motion-weight", type=float, default=1.0)
    parser.add_argument("--mapper", choices=["global", "incremental"], default="global")
    parser.add_argument(
        "--pose-convention",
        choices=["auto", "camera-to-world", "world-to-camera"],
        default="auto",
    )
    parser.add_argument("--output", type=Path, default=Path("reconstructions/property"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/lingbot-map-long.pt")
    )
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--overlap", type=int, default=24)
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    parser.add_argument(
        "--precision", choices=["float32", "bfloat16", "float16"], default="bfloat16"
    )
    args = parser.parse_args()
    artifact_output = args.output
    quality = None
    if not args.stage:
        print("description: Reconstruct observed property surfaces on this Mac")
        print('status: "Choose an input video to begin"')
        print(
            'help: "python -m lingbot_map.reconstruction run --video <video> --output <directory>"'
        )
        return
    try:
        if args.camera_source == "native" and args.stage in ("register", "repair"):
            raise ValueError("Native cameras use infer/fuse; run skips external registration with --camera-source native")
        if args.fps <= 0 or (args.limit is not None and args.limit < 1):
            raise ValueError("fps and limit must be positive")
        if args.stage in ("prepare", "run"):
            if args.images is not None:
                from .images import prepare_images

                prepare_images(args.images, args.output, args.limit)
            else:
                if args.video is None:
                    raise ValueError("--video or --images is required for prepare/run")
                from .io import prepare

                prepare(args.video, args.output, args.fps, args.limit)
        if args.stage in ("infer", "run"):
            with contextlib.redirect_stdout(sys.stderr):
                from .inference import infer

                infer(
                    args.output,
                    args.checkpoint,
                    args.window,
                    args.overlap,
                    args.device,
                    args.precision,
                    args.pose_convention,
                )
        if args.stage == "run" and args.camera_source != "native":
            from .repair import register_with_repairs
            from .sfm import reconstruct_cameras

            models = reconstruct_cameras(args.output, args.mapper)
            if len(models) != 1:
                raise ValueError(
                    f"Photogrammetry produced {len(models)} disconnected models; inspect them before assembling a building"
                )
            artifact_output = args.output / "final"
            artifact_output = register_with_repairs(
                args.output, artifact_output, models[0], args.checkpoint, args.bridges,
                camera_source=args.camera_source,
                motion_weight=args.motion_weight,
            )
        if args.stage in ("fuse", "run"):
            with contextlib.redirect_stdout(sys.stderr):
                from .fusion import fuse

                fuse(artifact_output)
        if args.stage in ("validate", "run"):
            from .validation import validate

            quality = validate(artifact_output)
            asset = next(
                name
                for name in (
                    "property-detail.glb",
                    "property-unlit.glb",
                    "property.glb",
                )
                if (artifact_output / "model" / name).exists()
            )
            web_quality = validate(artifact_output, asset_name=asset)
            quality["view_consistency_gate"] &= web_quality["view_consistency_gate"]
        if args.stage == "sfm":
            from .sfm import reconstruct_cameras

            reconstruct_cameras(args.output, args.mapper)
        if args.stage == "densify":
            if args.source is None:
                raise ValueError("densify requires --source and a new --output")
            from .densification import densify

            densify(args.source, args.output)
        if args.stage == "normalize":
            if args.source is None:
                raise ValueError("normalize requires --source and a new --output")
            from .normalization import normalize_archives

            normalize_archives(args.source, args.output)
        if args.stage == "refine":
            if args.source is None or args.colmap_model is None:
                raise ValueError(
                    "refine requires --source, --colmap-model and a new --output"
                )
            from .refinement import refine

            refine(args.source, args.output, args.colmap_model)
        if args.stage == "register":
            if args.source is None or args.colmap_model is None:
                raise ValueError(
                    "register requires --source, --colmap-model and --output"
                )
            from .registration import register

            register(
                args.source, args.output, args.colmap_model, args.bridges,
                camera_source=args.camera_source,
                motion_weight=args.motion_weight,
            )
        if args.stage == "repair":
            if args.source is None or args.colmap_model is None:
                raise ValueError(
                    "repair requires --source, --colmap-model and --output"
                )
            from .repair import register_with_repairs

            artifact_output = register_with_repairs(
                args.source,
                args.output,
                args.colmap_model,
                args.checkpoint,
                args.bridges,
                camera_source=args.camera_source,
                motion_weight=args.motion_weight,
            )
        if args.stage == "export":
            from .export import export_detail

            export_detail(args.output)
        if args.stage == "view":
            from .viewer import view

            view(args.output, port=args.port or 8081)
        if args.stage == "tour":
            if args.source is None:
                raise ValueError("tour requires --source and a new --output")
            from .tour import export_tour

            export_tour(args.source, args.output, args.stations, args.panorama_width, args.triangle_budget)
        if args.stage == "tour-view":
            from .tour_viewer import view_tour

            view_tour(args.output, port=args.port or 8083)
        if args.stage == "tour-sweep":
            if args.source is None:
                raise ValueError("tour-sweep requires --source tour and a new --output")
            from .panorama import export_sweep

            export_sweep(args.source, args.station, args.output,
                         mode=args.sweep_mode, frame_count=args.sweep_frames,
                         world_up=args.sweep_up)
        if args.stage == "tour-stitch":
            if args.source is None or args.images is None:
                raise ValueError("tour-stitch requires --source sweep, --images and a new --output")
            from .panorama import stitch_sweep

            stitch_sweep(args.source, args.images, args.output, args.panorama_width, args.generated)
        if args.stage == "tour-attach":
            if args.source is None:
                raise ValueError("tour-attach requires --source refinement and --output tour")
            from .panorama import attach_refinement

            attach_refinement(args.output, args.station, args.source)
        passed = quality is None or quality["view_consistency_gate"]
        print("status: complete" if passed else "status: needs_review")
        print("output: " + json.dumps(str(artifact_output.resolve())))
        if quality is not None:
            print("metric_accuracy_verified: false")
        if not passed:
            sys.exit(2)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print("error: " + json.dumps(str(error)))
        print(
            'help: "Check the input and configuration; saved windows can be resumed with infer"'
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
