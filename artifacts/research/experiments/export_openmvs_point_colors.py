"""Export unchanged native geometry with nearest dense-point colors for inspection."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

from lingbot_map.reconstruction.io import digest, write_json
from lingbot_map.reconstruction.materials import unlit_materials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing artifacts; choose a new --output")
    source, output = args.source_trial.resolve(), args.output.resolve()
    records = {}
    for stage, names in [("dense", ["dense.ply"]), ("mesh", ["mesh.ply", "mesh.mvs"])]:
        record = json.loads((source / f"{stage}-completion.json").read_text())
        if not record.get("complete"):
            raise ValueError(f"Source {stage} is incomplete")
        for name in names:
            records[name] = digest(source / name)
            if records[name] != record["outputs"][name]["sha256"]:
                raise ValueError(f"Source artifact changed: {name}")
    output.mkdir(parents=True)
    source_run = json.loads((source / "run.json").read_text())
    write_json(
        output / "run.json",
        {
            **source_run,
            "source_trial": str(source),
            "source_resources_sha256": records,
            "source_run_sha256": digest(source / "run.json"),
            "implementation_sha256": digest(Path(__file__)),
            "appearance": "Nearest dense-point vertex colors; no atlas or seam optimization",
        },
    )
    (output / "dense.ply").symlink_to(source / "dense.ply")
    began = time.monotonic()
    cloud = o3d.io.read_point_cloud(str(source / "dense.ply"))
    mesh = o3d.io.read_triangle_mesh(str(source / "mesh.ply"))
    if not cloud.has_colors() or not mesh.has_triangles():
        raise ValueError("Expected a colored dense point cloud and saved triangle mesh")
    distances, indices = cKDTree(np.asarray(cloud.points)).query(
        np.asarray(mesh.vertices), workers=6
    )
    colors = (
        (np.asarray(cloud.colors)[indices] * 255).round().clip(0, 255).astype(np.uint8)
    )
    asset = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=colors,
        process=False,
    )
    # Keep native COLMAP axes here; the shared app packager records its axis transform.
    asset.export(
        str(output / "property.glb"),
        file_type="glb",
        tree_postprocessor=unlit_materials,
    )
    recovered = trimesh.load_scene(
        output / "property.glb", process=False, skip_materials=True
    )
    part = next(iter(recovered.geometry.values()))
    if not np.array_equal(part.faces, asset.faces) or not np.array_equal(
        part.vertices, asset.vertices
    ):
        raise ValueError("Vertex-color export changed native geometry")
    if not np.array_equal(part.visual.vertex_colors[:, :3], colors):
        raise ValueError("Vertex-color export changed dense-point colors")
    if any(digest(source / name) != sha for name, sha in records.items()):
        raise ValueError("Native source changed during export")
    result = {
        "complete": True,
        "property_sha256": digest(output / "property.glb"),
        "seconds": time.monotonic() - began,
        "vertices": len(distances),
        "triangles": len(asset.faces),
        "exact_point_matches": int((distances == 0).sum()),
        "nearest_point_distance_quantiles_model_units": dict(
            zip(
                ["min", "p50", "p95", "p99", "max"],
                np.quantile(distances, [0, 0.5, 0.95, 0.99, 1]).tolist(),
            )
        ),
        "source_geometry_unchanged": True,
        "metric_accuracy_verified": False,
        "interpretation": "Observed dense-point colors transferred to unchanged mesh vertices. Nonzero nearest-point distances and vertex interpolation can blur appearance. This bypasses texture-atlas packing for geometry inspection; it does not repair camera errors, missing surfaces or scale.",
    }
    write_json(output / "completion.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
