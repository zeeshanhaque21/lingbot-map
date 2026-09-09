"""Run each supplied sequence locally, retaining logs and failed fidelity outcomes."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from lingbot_map.reconstruction.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--mapper", choices=("global", "incremental"), default="global")
    args = parser.parse_args()
    if args.records.exists():
        parser.error("Preserve existing run records; choose a new --records")
    args.records.mkdir(parents=True)
    for name in ("loop", "courthouse", "university"):
        command = [
            sys.executable,
            "-m",
            "lingbot_map.reconstruction",
            "run",
            "--images",
            str(args.examples.resolve() / name),
            "--output",
            str(args.output.resolve() / name),
            "--camera-source",
            "learned",
            "--motion-weight",
            "10",
            "--mapper",
            args.mapper,
        ]
        record = {
            "sequence": name,
            "command": command,
            "start_utc": datetime.now(timezone.utc).isoformat(),
        }
        began = time.monotonic()
        with (args.records / f"{name}.log").open("x") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            record["pid"] = process.pid
            write_json(args.records / f"{name}-running.json", record)
            print(json.dumps(record), flush=True)
            record["returncode"] = process.wait()
        record.update(
            {
                "seconds": time.monotonic() - began,
                "end_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        record["status"] = {0: "internal_screening_passed", 2: "needs_review"}.get(
            record["returncode"], "failed"
        )
        record["metric_accuracy_verified"] = False
        write_json(args.records / f"{name}-completion.json", record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
