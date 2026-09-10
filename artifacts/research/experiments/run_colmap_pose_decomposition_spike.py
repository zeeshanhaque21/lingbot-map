"""Test cached pair poses alone while preserving the original global-mapper inputs."""

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from lingbot_map.reconstruction.io import digest


def run_stage(output, name, command):
    started = time.monotonic()
    record = {"command": command, "start_utc": datetime.now(timezone.utc).isoformat()}
    with (output / f"{name}.log").open("x") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        record["pid"] = child.pid
        (output / f"{name}-started.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        print(json.dumps({"stage": name, **record}), flush=True)
        record["returncode"] = child.wait()
    record["seconds"] = time.monotonic() - started
    record["end_utc"] = datetime.now(timezone.utc).isoformat()
    (output / f"{name}-completion.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"stage": name, **record}), flush=True)
    if record["returncode"]:
        raise RuntimeError(f"Native {name} failed; inspect its preserved log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve earlier experiments; use a new output directory")
    args.database, args.images, args.project, args.output = (
        p.resolve() for p in (args.database, args.images, args.project, args.output)
    )
    wal = args.database.with_name(args.database.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError(
            "Source database has pending WAL data; use a consistent SQLite snapshot"
        )
    source_hash = digest(args.database)
    args.output.mkdir(parents=True)
    (args.output / "model").mkdir()
    shutil.copy2(__file__, args.output / "producer.py")
    snapshot = args.output / "database.db"
    shutil.copy2(args.database, snapshot)
    if (
        digest(snapshot) != source_hash
        or digest(args.database) != source_hash
        or (wal.exists() and wal.stat().st_size)
    ):
        raise ValueError("Database changed during snapshot creation")
    binary = shutil.which("colmap")
    if binary is None:
        raise ValueError("COLMAP is unavailable")
    # Generate a separate native config; all unrelated original options stay byte-identical.
    lines = args.project.read_text().splitlines()
    replacements = {
        "database_path": str(snapshot),
        "image_path": str(args.images),
        "output_path": str(args.output / "model"),
        "decompose_relative_pose": "false",
    }
    changed = set()
    for index, line in enumerate(lines):
        key = line.split("=", 1)[0]
        if key in replacements:
            lines[index] = key + "=" + replacements[key]
            changed.add(key)
    if changed != set(replacements):
        raise ValueError("Original mapper project is missing expected settings")
    project = args.output / "experiment.ini"
    project.write_text("\n".join(lines) + "\n")
    signature = {
        "source_database_sha256": source_hash,
        "snapshot_database_sha256": digest(snapshot),
        "source_project_sha256": digest(args.project),
        "experiment_project_sha256": digest(project),
        "binary_sha256": digest(Path(binary).resolve()),
        "producer_sha256": digest(Path(__file__)),
        "changed_algorithm_option": {"GlobalMapper.decompose_relative_pose": False},
        "interpretation": "Separate, byte-verified database copy and otherwise unchanged native global-mapper project. Tests whether adding decomposed poses to the cached calibrated pair graph explains the disputed turns. No original cameras, database or mesh are modified. This is a hypothesis test, not a fidelity claim.",
    }
    (args.output / "run.json").write_text(json.dumps(signature, indent=2) + "\n")
    run_stage(
        args.output, "mapper", [binary, "global_mapper", "--project_path", str(project)]
    )
    models = []
    for model in sorted((args.output / "model").iterdir()):
        if not model.is_dir() or not model.name.isdigit():
            continue
        text = model / "text"
        text.mkdir()
        run_stage(
            args.output,
            "convert-" + model.name,
            [
                binary,
                "model_converter",
                "--input_path",
                str(model),
                "--output_path",
                str(text),
                "--output_type",
                "TXT",
            ],
        )
        models.append(
            {
                "name": model.name,
                "text_sha256": {
                    name: digest(text / name)
                    for name in ("cameras.txt", "images.txt", "points3D.txt")
                },
            }
        )
    if not models:
        raise ValueError("Native mapper produced no reconstruction")
    result = {
        "complete": True,
        "models": models,
        "original_database_unchanged": digest(args.database) == source_hash
        and not (wal.exists() and wal.stat().st_size),
        "metric_accuracy_verified": False,
    }
    (args.output / "completion.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
