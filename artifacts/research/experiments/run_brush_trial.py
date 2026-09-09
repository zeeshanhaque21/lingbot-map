"""Run a bounded native Brush trial with durable logs and checkpoints."""

import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

from artifacts.research.experiments.mapanything_mac import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--max-splats", type=int, default=500000)
    parser.add_argument("--growth-stop", type=int, default=2000)
    parser.add_argument("--max-resolution", type=int, default=518)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing trials; use a new output directory")
    if min(args.steps, args.max_splats, args.growth_stop, args.max_resolution) < 1:
        parser.error("Trial limits must be positive")
    if args.growth_stop > args.steps:
        parser.error("Growth must stop no later than the final step")
    if not (args.dataset / "provenance.json").is_file():
        parser.error("Dataset preparation is incomplete")
    output = args.output.resolve()
    command = [
        str(args.binary.resolve()),
        str(args.dataset.resolve()),
        "--total-steps",
        str(args.steps),
        "--max-splats",
        str(args.max_splats),
        "--max-resolution",
        str(args.max_resolution),
        "--eval-every",
        "500",
        "--eval-save-to-disk",
        "--export-every",
        "500",
        "--export-path",
        str(output),
        "--refine-every",
        "100",
        "--growth-stop-iter",
        str(args.growth_stop),
        "--seed",
        "42",
    ]
    record = {
        "args": command,
        "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_provenance_sha256": digest(args.dataset / "provenance.json"),
        "binary_sha256": digest(args.binary),
        "RUST_LOG": "info",
        "metric_accuracy_verified": False,
    }
    output.mkdir(parents=True)
    (output / "run.json").write_text(json.dumps(record, indent=2))
    start = time.monotonic()
    with (output / "train.log").open("x") as log:
        child = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "RUST_LOG": "info"},
            start_new_session=True,
        )
        (output / "pid.json").write_text(
            json.dumps({"pid": child.pid, "args": command}, indent=2)
        )
        print(json.dumps({"pid": child.pid, "output": str(output)}), flush=True)
        code = child.wait()
    result = {
        "returncode": code,
        "seconds": time.monotonic() - start,
        "end_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (output / "completion.json").write_text(json.dumps(result, indent=2))
    if code:
        raise SystemExit(code)
    exports = list(output.glob("export_*.ply"))
    if not exports or not (output / f"eval_{args.steps}").is_dir():
        raise RuntimeError(
            "Brush exited successfully without expected evaluation or model artifacts"
        )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
