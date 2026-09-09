"""Package the complete native mesh and compare it with captured reserved views."""

import argparse
import json
import shutil
import struct
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import open3d as o3d
import trimesh

from artifacts.research.experiments.gaussian_depth_renderer import digest
from lingbot_map.reconstruction.colmap_io import read_model
from lingbot_map.reconstruction.validation import validate


def glb_document(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data)
    size, kind = struct.unpack_from("<II", data, 12)
    if magic != b"glTF" or version != 2 or length != len(data) or kind != 0x4E4F534A:
        raise ValueError("Invalid GLB container")
    return json.loads(data[20 : 20 + size])


def sparse_audit(mesh, colmap_text, cameras):
    images, points = read_model(colmap_text)
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(
        o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(np.asarray(mesh.faces), dtype=o3d.core.Dtype.UInt32),
    )
    rows = []
    for frame in cameras:
        number = frame["frame"] + 760
        image = images[f"{number:06d}.jpg"]
        camera = image["camera"]
        k, extrinsic = camera["intrinsics"], image["extrinsics"]
        c2w = np.linalg.inv(np.vstack([extrinsic, [0, 0, 0, 1]]))
        pixels, reference_z = [], []
        for x, y, identifier in image["observations"]:
            point = points.get(int(identifier))
            if point is None or point["track_length"] < 3:
                continue
            position = extrinsic[:, :3] @ point["xyz"] + extrinsic[:, 3]
            if position[2] <= 0:
                continue
            projected = k @ position
            if np.linalg.norm(projected[:2] / projected[2] - [x, y]) > 2:
                continue
            if not (
                0.5 <= x < camera["width"] - 0.5 and 0.5 <= y < camera["height"] - 0.5
            ):
                continue
            pixels.append([x, y, 1])
            reference_z.append(position[2])
        if not pixels:
            raise ValueError("No usable sparse observations")
        directions = np.asarray(pixels) @ np.linalg.inv(k).T @ c2w[:3, :3].T
        rays = np.column_stack(
            [np.broadcast_to(c2w[:3, 3], directions.shape), directions]
        ).astype(np.float32)
        hit = ray_scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        visible = np.isfinite(hit)
        predicted_z = hit * (directions @ extrinsic[2, :3])
        error = np.abs(predicted_z - reference_z) / reference_z
        rows.append(
            {
                "frame": number,
                "observations": len(pixels),
                "coverage": float(visible.mean()),
                "within_5percent_all_tracks": float(np.mean(visible & (error < 0.05))),
                "median_relative_error_covered": float(np.median(error[visible]))
                if visible.any()
                else None,
                "p95_relative_error_covered": float(np.quantile(error[visible], 0.95))
                if visible.any()
                else None,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--colmap-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing comparisons; use a new output directory")
    completion = json.loads((args.trial / "completion.json").read_text())
    source = args.trial / "property.glb"
    if not completion["complete"] or digest(source) != completion["property_sha256"]:
        raise ValueError("Native output is incomplete or changed")
    source_document = glb_document(source)
    resources = {"property.glb": digest(source)}
    for item in source_document.get("images", []) + source_document.get("buffers", []):
        uri = item.get("uri")
        if not uri or uri.startswith("data:"):
            continue
        resource = (args.trial / unquote(uri)).resolve()
        if not resource.is_relative_to(args.trial.resolve()):
            raise ValueError("Native asset refers outside its artifact directory")
        resources[uri] = digest(resource)
    scene = trimesh.load_scene(source, process=False)
    if len(scene.geometry) != 1 or len(scene.graph.nodes_geometry) != 1:
        raise ValueError(
            "This comparison requires one texture material; do not silently merge materials"
        )
    native = scene.to_mesh()
    if getattr(native.visual.material, "baseColorTexture", None) is None:
        raise ValueError("Native texture did not load")
    root = args.output / "model"
    root.mkdir(parents=True)
    alignment = json.loads((args.reference / "model/alignment.json").read_text())
    if not all(
        np.allclose(transform, np.eye(4), atol=1e-7, rtol=0)
        for transform in alignment["transforms"]
    ):
        raise ValueError("Reference must use the unchanged COLMAP world coordinates")
    cameras = json.loads((args.reference / "model/cameras.json").read_text())
    if len(cameras) != 48 or [
        camera["frame"] for camera in cameras if camera.get("held_out_from_fusion")
    ] != [5, 15, 25, 35, 45]:
        raise ValueError("Expected the calibrated 760-807 chair-section reference")
    axis = np.diag([1.0, -1.0, -1.0, 1.0])
    packed = native.copy()
    packed.apply_transform(axis)
    packed.export(root / "property.glb")
    document = glb_document(root / "property.glb")
    if any(
        "uri" in item
        for item in document.get("images", []) + document.get("buffers", [])
    ):
        raise ValueError("Packed GLB still needs external resources")
    recovered = trimesh.load(root / "property.glb", force="mesh", process=False)
    recovered.apply_transform(axis)
    if not np.array_equal(recovered.faces, native.faces) or not np.allclose(
        recovered.vertices, native.vertices, atol=1e-7, rtol=0
    ):
        raise ValueError("Packaging changed complete mesh geometry")
    if not np.array_equal(
        np.asarray(recovered.visual.material.baseColorTexture),
        np.asarray(native.visual.material.baseColorTexture),
    ) or not np.allclose(recovered.visual.uv, native.visual.uv, atol=1e-7, rtol=0):
        raise ValueError("Packaging changed texture pixels or coordinates")
    for name in ("cameras.json", "alignment.json"):
        shutil.copy2(args.reference / "model" / name, root / name)
    (args.output / "windows").symlink_to(
        (args.reference / "windows").resolve(), target_is_directory=True
    )
    report = validate(args.output, asset_name="property.glb")
    sparse = sparse_audit(
        native,
        args.colmap_text,
        [camera for camera in cameras if camera.get("held_out_from_fusion")],
    )
    result = {
        "source_resources_sha256": resources,
        "property_sha256": digest(root / "property.glb"),
        "property_bytes": (root / "property.glb").stat().st_size,
        "triangles": len(recovered.faces),
        "texture_size": list(native.visual.material.baseColorTexture.size),
        "reference_files_sha256": {
            name: digest(args.reference / "model" / name)
            for name in ("cameras.json", "alignment.json", "web-render-validation.json")
        },
        "reference_windows_sha256": {
            path.name: digest(path)
            for path in sorted((args.reference / "windows").glob("*.npz"))
        },
        "colmap_sha256": {
            name: digest(args.colmap_text / name)
            for name in ("cameras.txt", "images.txt", "points3D.txt")
        },
        "reserved_view_comparison": report,
        "gaussian_mesh_baseline": json.loads(
            (args.reference / "model/web-render-validation.json").read_text()
        ),
        "sparse_observations": sparse,
        "metric_accuracy_verified": False,
        "ready_for_verified_property_listing": False,
        "interpretation": "Complete textured mesh, without triangle reduction. Captured RGB and cameras match the higher-resolution Gaussian mesh comparison. Dense depth-support references remain Gaussian-derived, so they measure model agreement, not ground truth. Sparse COLMAP observations participated in camera estimation and are not independent surveyed measurements. Native OpenMVS world coordinates are preserved through a recorded GLB axis conversion. External native textures are embedded losslessly in the final GLB.",
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "triangles": result["triangles"],
                "bytes": result["property_bytes"],
                "coverage": report["median_rendered_coverage"],
                "depth_support": report["median_supported_depth_fraction"],
                "visible_rgb_error": report["median_absolute_rgb_error_visible"],
                "screening_gate": report["view_consistency_gate"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
