"""Run with python -m lingbot_map.reconstruction."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="Reconstruct observed property surfaces on this Mac")
    parser.add_argument("stage", choices=["prepare", "infer", "run", "fuse", "view"], nargs="?")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reconstructions/property"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/lingbot-map-long.pt"))
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--overlap", type=int, default=24)
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    parser.add_argument("--precision", choices=["float32", "bfloat16", "float16"], default="float32")
    args = parser.parse_args()
    if not args.stage:
        print('description: Reconstruct observed property surfaces on this Mac')
        print('status: "Choose an input video to begin"')
        print('help: "python -m lingbot_map.reconstruction run --video <video> --output <directory>"')
        return
    try:
        if args.fps <= 0 or (args.limit is not None and args.limit < 1):
            raise ValueError("fps and limit must be positive")
        if args.stage in ("prepare", "run"):
            if args.video is None:
                raise ValueError("--video is required for prepare/run")
            from .io import prepare
            prepare(args.video, args.output, args.fps, args.limit)
        if args.stage in ("infer", "run"):
            from .inference import infer
            infer(args.output, args.checkpoint, args.window, args.overlap, args.device, args.precision)
        if args.stage in ("fuse", "run"):
            from .fusion import fuse
            fuse(args.output)
        if args.stage == "view":
            from .viewer import view
            view(args.output)
        print('status: complete')
        print('output: ' + json.dumps(str(args.output.resolve())))
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print('error: ' + json.dumps(str(error)))
        print('help: "Check the input and configuration; saved windows can be resumed with infer"')
        sys.exit(1)


if __name__ == "__main__":
    main()
