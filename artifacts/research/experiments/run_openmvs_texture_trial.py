"""Retexture a verified native mesh without repeating stereo or changing its geometry."""

import argparse
import json
import shutil
from pathlib import Path

from artifacts.research.experiments.run_colmap_pose_decomposition_spike import run_stage
from lingbot_map.reconstruction.io import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trial", type=Path, required=True)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--virtual-face-images", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing native runs; choose a new output directory")
    if args.virtual_face_images < 0:
        parser.error("The virtual-face view count must be nonnegative")
    args.source_trial, args.binaries, args.output = (
        p.resolve() for p in (args.source_trial, args.binaries, args.output)
    )
    source_run = json.loads((args.source_trial / "run.json").read_text())
    mesh_stage = json.loads((args.source_trial / "mesh-completion.json").read_text())
    mesh = args.source_trial / "mesh.mvs"
    if (
        not mesh_stage.get("complete")
        or digest(mesh) != mesh_stage["outputs"]["mesh.mvs"]["sha256"]
    ):
        raise ValueError("Source mesh is incomplete or changed")
    binary = args.binaries / "TextureMesh"
    if digest(binary) != source_run["binary_sha256"]["TextureMesh"]:
        raise ValueError("Use the same verified texturing binary for this comparison")
    if source_run["seam_leveling"] != "off":
        raise ValueError(
            "Compare against the unchanged mesh with seam adjustments disabled"
        )
    args.output.mkdir(parents=True)
    shutil.copy2(__file__, args.output / "producer.py")
    # Inputs are immutable artifacts; all native outputs go to this new directory.
    (args.output / "dense.ply").symlink_to((args.source_trial / "dense.ply").resolve())
    run = {
        **source_run,
        "source_trial": str(args.source_trial),
        "source_mesh_sha256": digest(mesh),
        "source_run_sha256": digest(args.source_trial / "run.json"),
        "virtual_face_images": args.virtual_face_images,
        "implementation_sha256": digest(Path(__file__)),
        "interpretation": "Texture assignment and packing comparison on the identical saved native mesh. Coplanar triangles may share a view assignment; no geometry reduction, hole filling, seam adjustment or sharpening is requested. Building geometry remains unverified.",
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    command = [
        str(binary),
        "--input-file",
        str(mesh),
        "--output-file",
        str(args.output / "property.mvs"),
        "--working-folder",
        str(args.output),
        "--export-type",
        "glb",
        "--close-holes",
        "0",
        "--sharpness-weight",
        "0",
        "--global-seam-leveling",
        "0",
        "--local-seam-leveling",
        "0",
        "--virtual-face-images",
        str(args.virtual_face_images),
        "--max-threads",
        str(source_run["threads"]),
        "--archive-type",
        "2",
    ]
    run_stage(args.output, "texture", command)
    result = {
        "complete": True,
        "property_sha256": digest(args.output / "property.glb"),
        "source_mesh_unchanged": digest(mesh) == run["source_mesh_sha256"],
        "metric_accuracy_verified": False,
    }
    (args.output / "completion.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
