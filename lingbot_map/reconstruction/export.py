"""Export app geometry with a budget proportional to captured scene extent."""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from .io import digest, write_json
from .materials import unlit_materials


def simplify_for_app(mesh, voxel, frame_count):
    target = max(300000, frame_count * 3000)
    if len(mesh.triangles) > target:
        # Bound the first reduction by reconstruction resolution before quadric fitting.
        mesh = mesh.simplify_vertex_clustering(voxel_size=voxel)
        if len(mesh.triangles) > target:
            mesh = mesh.simplify_quadric_decimation(target)
    mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.clip(np.asarray(mesh.vertex_colors), 0, 1)
    )
    return mesh


def export_detail(output):
    root = Path(output) / "model"
    report = json.loads((root / "validation.json").read_text())
    source_hash = digest(root / "observed-surfaces.ply")
    marker, target = root / "app-export.json", root / "property-detail.glb"
    if marker.exists():
        prior = json.loads(marker.read_text())
        if (
            prior["source_mesh_sha256"] == source_hash
            and target.exists()
            and digest(target) == prior["output_sha256"]
        ):
            return
        raise ValueError(
            "Detailed export cache changed; preserve it and use a new output"
        )
    mesh = o3d.io.read_triangle_mesh(str(root / "observed-surfaces.ply"))
    mesh = simplify_for_app(
        mesh, report["voxel_size_model_units"], report["frames_expected"]
    )
    vertices = np.asarray(mesh.vertices).copy()
    vertices[:, 1:] *= -1
    asset = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(mesh.triangles),
        vertex_colors=(np.asarray(mesh.vertex_colors) * 255).round().astype(np.uint8),
        process=False,
    )
    asset.metadata.update(
        units="uncalibrated model units",
        evidence="Observed surface mesh",
        metric_accuracy="unverified",
    )
    asset.export(target, tree_postprocessor=unlit_materials)
    write_json(
        marker,
        {
            "source_mesh_sha256": source_hash,
            "output_sha256": digest(target),
            "triangles": len(mesh.triangles),
            "vertices": len(mesh.vertices),
            "voxel_size": report["voxel_size_model_units"],
            "method": "Resolution-bounded vertex clustering followed by quadric simplification",
            "metric_accuracy_verified": False,
        },
    )
