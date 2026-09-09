"""Run a recorded native CPU multiview reconstruction and export a textured GLB."""

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from artifacts.research.experiments.gaussian_depth_renderer import digest


def dense_handoff(path):
    """Check the native PLY metadata required by visibility-weighted meshing."""
    with path.open("rb") as handle:
        header = handle.read(4096).split(b"end_header", 1)[0].decode("ascii")
    return {
        "has_view_indices": "property list uint8 uint32 view_indices" in header,
        "has_view_weights": "property list uint8 float32 view_weights" in header,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seam-leveling", choices=["on", "off"], default="on")
    parser.add_argument(
        "--reuse-dense",
        type=Path,
        help="Reuse verified import/dense artifacts from a previous trial",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing runs; use a new output directory")
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.binaries, args.dataset, args.output = (
        path.resolve() for path in (args.binaries, args.dataset, args.output)
    )
    provenance = json.loads((args.dataset / "provenance.json").read_text())
    for name, expected in provenance["image_sha256"].items():
        if digest(args.dataset / "images" / name) != expected:
            raise ValueError("Captured training image changed")
    for name, expected in provenance["exported_colmap_sha256"].items():
        if digest(args.dataset / "sparse" / name) != expected:
            raise ValueError("Calibrated sparse input changed")
    stages = [
        (
            "import",
            "InterfaceCOLMAP",
            [
                "--input-file",
                str(args.dataset),
                "--image-folder",
                str(args.dataset / "images"),
                "--output-file",
                "scene.mvs",
            ],
            ["scene.mvs"],
        ),
        (
            "dense",
            "DensifyPointCloud",
            [
                "--input-file",
                "scene.mvs",
                "--output-file",
                "dense.mvs",
                "--resolution-level",
                "0",
                "--max-resolution",
                "1280",
                "--number-views",
                "5",
                "--geometric-iters",
                "2",
                "--number-views-fuse",
                "2",
                "--estimate-roi",
                "0",
                "--crop-to-roi",
                "0",
                "--tower-mode",
                "0",
                "--remove-dmaps",
                "0",
            ],
            ["dense.mvs", "dense.ply"],
        ),
        (
            "mesh",
            "ReconstructMesh",
            [
                "--input-file",
                "dense.mvs",
                "--pointcloud-file",
                "dense.ply",
                "--output-file",
                "mesh.mvs",
                "--close-holes",
                "0",
                "--smooth",
                "0",
                "--crop-to-roi",
                "0",
            ],
            ["mesh.mvs", "mesh.ply"],
        ),
        (
            "texture",
            "TextureMesh",
            [
                "--input-file",
                "mesh.mvs",
                "--output-file",
                "property.mvs",
                "--export-type",
                "glb",
                "--close-holes",
                "0",
                "--sharpness-weight",
                "0",
                "--global-seam-leveling",
                "1" if args.seam_leveling == "on" else "0",
                "--local-seam-leveling",
                "1" if args.seam_leveling == "on" else "0",
            ],
            ["property.mvs", "property.glb"],
        ),
    ]
    binaries = {name: digest(args.binaries / name) for _, name, _, _ in stages}
    reuse = {}
    if args.reuse_dense:
        args.reuse_dense = args.reuse_dense.resolve()
        previous = json.loads((args.reuse_dense / "run.json").read_text())
        if previous["dataset_provenance_sha256"] != digest(
            args.dataset / "provenance.json"
        ):
            raise ValueError(
                "Reused dense reconstruction belongs to different input data"
            )
        for stage, binary, options, expected_outputs in stages[:2]:
            record = json.loads(
                (args.reuse_dense / f"{stage}-completion.json").read_text()
            )
            if (
                not record["complete"]
                or record["returncode"] != 0
                or previous["binary_sha256"][binary] != binaries[binary]
            ):
                raise ValueError(
                    "Reused stage is incomplete or uses a different binary"
                )
            expected_options = [*options, "--max-threads", str(args.threads)]
            if record["command"][1:] not in [
                expected_options,
                [*expected_options, "--archive-type", "-1"],
            ]:
                raise ValueError("Reused reconstruction parameters differ")
            for name in expected_outputs:
                if digest(args.reuse_dense / name) != record["outputs"][name]["sha256"]:
                    raise ValueError("Reused output changed")
            reuse[stage] = record
        if not all(dense_handoff(args.reuse_dense / "dense.ply").values()):
            raise ValueError(
                "Reused dense PLY lacks per-point view metadata required for meshing"
            )
    args.output.mkdir(parents=True)
    shutil.copy2(Path(__file__), args.output / "producer.py")
    (args.output / "run.json").write_text(
        json.dumps(
            {
                "start_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_provenance_sha256": digest(args.dataset / "provenance.json"),
                "binary_sha256": binaries,
                "implementation_sha256": digest(Path(__file__)),
                "threads": args.threads,
                "seam_leveling": args.seam_leveling,
                "reuse_dense": str(args.reuse_dense) if args.reuse_dense else None,
                "metric_accuracy_verified": False,
                "interpretation": "Classical CPU multiview stereo on captured training RGB. Artificial tower points, automatic region cropping, hole filling, mesh smoothing and texture sharpening are disabled. The surface still requires reserved-view and independent property checks.",
            },
            indent=2,
        )
    )
    for stage, binary, options, expected_outputs in stages:
        if stage in reuse:
            for name in expected_outputs:
                shutil.copy2(args.reuse_dense / name, args.output / name)
            record = {
                "stage": stage,
                "reused_from": str(args.reuse_dense),
                "source_completion_sha256": digest(
                    args.reuse_dense / f"{stage}-completion.json"
                ),
                "outputs": reuse[stage]["outputs"],
                "complete": True,
            }
            (args.output / f"{stage}-reused.json").write_text(
                json.dumps(record, indent=2)
            )
            print(json.dumps(record), flush=True)
            continue
        command = [
            str(args.binaries / binary),
            *options,
            "--max-threads",
            str(args.threads),
            "--archive-type",
            "-1" if stage in ("import", "dense") else "2",
        ]
        record = {
            "stage": stage,
            "command": command,
            "start_utc": datetime.now(timezone.utc).isoformat(),
        }
        start = time.monotonic()
        with (args.output / f"{stage}.log").open("xb") as log:
            process = subprocess.Popen(
                command,
                cwd=args.output,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record["pid"] = process.pid
            (args.output / f"{stage}-started.json").write_text(
                json.dumps(record, indent=2)
            )
            print(json.dumps(record), flush=True)
            returncode = process.wait()
        record.update(
            returncode=returncode,
            seconds=time.monotonic() - start,
            end_utc=datetime.now(timezone.utc).isoformat(),
        )
        record["outputs"] = {
            name: {
                "bytes": (args.output / name).stat().st_size,
                "sha256": digest(args.output / name),
            }
            for name in expected_outputs
            if (args.output / name).is_file()
        }
        record["complete"] = returncode == 0 and len(record["outputs"]) == len(
            expected_outputs
        )
        if stage == "dense" and record["complete"]:
            record["dense_handoff"] = dense_handoff(args.output / "dense.ply")
            record["complete"] = all(record["dense_handoff"].values())
        (args.output / f"{stage}-completion.json").write_text(
            json.dumps(record, indent=2)
        )
        print(json.dumps(record), flush=True)
        if not record["complete"]:
            raise RuntimeError(
                f"OpenMVS {stage} failed or omitted an expected artifact; inspect {stage}.log"
            )
    (args.output / "completion.json").write_text(
        json.dumps(
            {
                "complete": True,
                "property_sha256": digest(args.output / "property.glb"),
                "metric_accuracy_verified": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
