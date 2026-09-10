"""Verify additional end-to-start image pairs in an isolated copy of a capture."""

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path

from artifacts.research.experiments.run_colmap_pose_decomposition_spike import run_stage
from lingbot_map.reconstruction.io import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end-window", type=int, default=20)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or args.end_window < 2:
        parser.error("Use a fresh output and an end window of at least two images")
    manifest = json.loads((source / "input.json").read_text())
    frames = manifest["frames"]
    if (
        manifest["configuration"].get("source_kind") != "image_sequence"
        or len(frames) < args.end_window * 2
    ):
        raise ValueError("Expected an image sequence longer than both matching windows")
    database = source / "colmap/database.db"
    source_hash = digest(database)
    (output / "colmap").mkdir(parents=True)
    for name in ["input.json", "inference.json"]:
        shutil.copy2(source / name, output / name)
    for name in ["frames", "windows"]:
        (output / name).symlink_to(source / name, target_is_directory=True)
    for name in [
        "configuration.json",
        "features.complete.json",
        "matching.complete.json",
    ]:
        shutil.copy2(source / "colmap" / name, output / "colmap" / name)
    with (
        sqlite3.connect(f"file:{database}?mode=ro", uri=True) as original,
        sqlite3.connect(output / "colmap/database.db") as copied,
    ):
        original.backup(copied)
    pairs = [
        f"{Path(a['file']).name} {Path(b['file']).name}"
        for a in frames[: args.end_window]
        for b in frames[-args.end_window :]
    ]
    pair_file = output / "closure-pairs.txt"
    pair_file.write_text("\n".join(pairs) + "\n")
    write_json(
        output / "closure-input.json",
        {
            "source": str(source),
            "source_database_sha256": source_hash,
            "pairs_sha256": digest(pair_file),
            "proposed_pairs": len(pairs),
            "interpretation": "Additional end-to-start candidate pairs require native feature and geometric verification. No camera coincidence or loop transform is imposed. Existing windows and baseline outputs remain unchanged.",
        },
    )
    run_stage(
        output,
        "closure-matching",
        [
            "colmap",
            "matches_importer",
            "--database_path",
            str(output / "colmap/database.db"),
            "--match_list_path",
            str(pair_file),
            "--match_type",
            "pairs",
            "--FeatureMatching.use_gpu",
            "0",
            "--FeatureMatching.num_threads",
            "6",
        ],
    )
    if digest(database) != source_hash:
        raise ValueError("Baseline feature database changed during closure experiment")
    run_stage(
        output,
        "pipeline",
        [
            sys.executable,
            "-m",
            "lingbot_map.reconstruction",
            "run",
            "--images",
            manifest["configuration"]["source"],
            "--output",
            str(output),
            "--camera-source",
            "learned",
            "--motion-weight",
            "10",
            "--mapper",
            "global",
        ],
    )


if __name__ == "__main__":
    main()
