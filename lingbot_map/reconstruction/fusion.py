"""Fuse only multi-view-supported depth into a colored, open surface mesh."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from .geometry import align_overlap, supported_depth
from .io import digest, write_json


def fuse(output):
    output = Path(output)
    files = sorted((output / "windows").glob("*.npz"))
    if not files:
        raise ValueError("No inference windows; run infer first")
    manifest = json.loads((output / "input.json").read_text())
    inference = json.loads((output / "inference.json").read_text())
    expected = len(manifest["frames"])
    from .inference import window_ranges

    ranges = list(window_ranges(expected, inference["window"], inference["overlap"]))
    if [int(p.stem) for p in files] != [start for start, _ in ranges]:
        raise ValueError("Incomplete inference; resume infer before final fusion")
    artifact = output / "model"
    input_signature = {
        "inference": digest(output / "inference.json"),
        "windows": {p.name: digest(p) for p in files},
    }
    if (artifact / "validation.json").exists():
        saved = json.loads((artifact / "validation.json").read_text())
        if (
            saved["source_sha256"] != manifest["configuration"]["source_sha256"]
            or saved["frames_expected"] != expected
            or saved.get("input_signature") != input_signature
        ):
            raise ValueError(
                "Existing model belongs to different input; use a new output directory"
            )
        if not all(
            (artifact / name).exists() and digest(artifact / name) == checksum
            for name, checksum in saved["files"].items()
        ):
            raise ValueError(
                "Existing model files are missing or changed; preserve them and use a new output directory"
            )
        return
    artifact.mkdir(exist_ok=True)
    transforms, registrations = [np.eye(4)], []
    previous = dict(np.load(files[0]))
    reference_depth = previous["depth"]
    if "registered" in previous:
        reference_depth = reference_depth[previous["registered"]]
        if not reference_depth.size:
            for path in files[1:]:
                candidate = dict(np.load(path))
                reference_depth = candidate["depth"][candidate["registered"]]
                if reference_depth.size:
                    break
        if not reference_depth.size:
            raise ValueError("No frames passed external depth calibration")
    voxel = float(np.median(reference_depth) / 180)
    for path in files[1:]:
        current = dict(np.load(path))
        if inference.get("poses_global"):
            transform, stats = (
                np.eye(4),
                {"method": "shared external camera coordinates"},
            )
        else:
            transform, stats = align_overlap(previous, current)
        transforms.append(transforms[-1] @ transform)
        registrations.append(stats)
        previous = current
    del previous
    write_json(
        artifact / "alignment.json",
        {
            "transforms": [t.tolist() for t in transforms],
            "overlaps": registrations,
            "method": inference.get(
                "pose_source", "robust same-pixel Sim3 on shared frames"
            ),
            "loop_closure": False,
        },
    )
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=voxel * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    diagnostics, cameras = [], []
    started = time.monotonic()
    for chunk, (path, transform) in enumerate(zip(files, transforms)):
        data = dict(np.load(path))
        scale = float(np.cbrt(np.linalg.det(transform[:3, :3])))
        rotation = transform[:3, :3] / scale
        start, end = ranges[chunk]
        # Prefer predictions after the scale bootstrap; use each frame exactly once.
        low = start if chunk == 0 else (start + ranges[chunk - 1][1]) // 2
        high = end if chunk == len(files) - 1 else (end + ranges[chunk + 1][0]) // 2
        for i, frame_id in enumerate(data["frame_ids"]):
            if not low <= frame_id < high:
                continue
            if "registered" in data and not data["registered"][i]:
                continue
            filtered, stats = supported_depth(i, data)
            filtered *= scale
            extrinsic = data["extrinsics"][i].astype(np.float64)
            local_c2w = np.eye(4)
            local_c2w[:3, :3] = extrinsic[:, :3].T
            local_c2w[:3, 3] = -extrinsic[:, :3].T @ extrinsic[:, 3]
            global_c2w = np.eye(4)
            global_c2w[:3, :3] = rotation @ local_c2w[:3, :3]
            global_c2w[:3, 3] = scale * rotation @ local_c2w[:3, 3] + transform[:3, 3]
            k = data["intrinsics"][i]
            h, w = filtered.shape
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                w, h, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
            )
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(data["rgb"][i])),
                o3d.geometry.Image(filtered),
                depth_scale=1.0,
                depth_trunc=float(filtered.max() + 1),
                convert_rgb_to_intensity=False,
            )
            held_out = int(frame_id) % 10 == 5
            if not held_out:
                volume.integrate(rgbd, intrinsic, np.linalg.inv(global_c2w))
            stats["held_out_from_fusion"] = held_out
            cameras.append(
                {
                    "frame": int(frame_id),
                    "timestamp_seconds": manifest["frames"][int(frame_id)][
                        "timestamp_seconds"
                    ],
                    "camera_to_world": global_c2w.tolist(),
                    "intrinsics": k.tolist(),
                    "window": chunk,
                    "held_out_from_fusion": held_out,
                }
            )
            diagnostics.append(stats)
        write_json(
            artifact / "progress.json",
            {"windows_fused": chunk + 1, "frames_fused": len(cameras)},
        )
        print(f"Fused {len(cameras)}/{expected} frames", file=sys.stderr, flush=True)
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.clip(np.asarray(mesh.vertex_colors), 0, 1)
    )
    if len(mesh.triangles) == 0:
        raise ValueError(
            "No supported surfaces survived fusion; inspect depth/pose diagnostics"
        )
    o3d.io.write_triangle_mesh(str(artifact / "observed-surfaces.ply"), mesh)
    cloud = volume.extract_point_cloud()
    cloud.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(cloud.colors), 0, 1))
    o3d.io.write_point_cloud(str(artifact / "observed-points.ply"), cloud)
    from .export import simplify_for_app

    web_mesh = simplify_for_app(mesh, voxel, expected)
    vertices = np.asarray(web_mesh.vertices).copy()
    # glTF is Y-up; rotate OpenCV world Y/Z to match. The source PLY stays OpenCV.
    vertices[:, 1:] *= -1
    asset = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(web_mesh.triangles),
        vertex_colors=(np.clip(np.asarray(web_mesh.vertex_colors), 0, 1) * 255)
        .round()
        .astype(np.uint8),
        process=False,
    )
    asset.metadata.update(
        {
            "units": "uncalibrated model units",
            "evidence": "multi-view supported depth",
            "metric_accuracy": "unverified",
            "unseen_surfaces": "not completed",
        }
    )
    from .materials import unlit_materials

    asset.export(artifact / "property.glb", tree_postprocessor=unlit_materials)
    write_json(artifact / "cameras.json", cameras)
    write_json(artifact / "frame-validation.json", diagnostics)
    report = {
        "status": "reconstructed_unverified_metric_accuracy",
        "source_sha256": manifest["configuration"]["source_sha256"],
        "input_signature": input_signature,
        "frames_expected": expected,
        "frames_processed": len(cameras),
        "frames_fused": sum(not c["held_out_from_fusion"] for c in cameras),
        "frames_held_out": sum(c["held_out_from_fusion"] for c in cameras),
        "temporal_coverage": len(cameras) / expected,
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "web_triangles": len(web_mesh.triangles),
        "voxel_size_model_units": voxel,
        "units": "uncalibrated model units",
        "accepted_pixel_fraction": sum(x["accepted_pixels"] for x in diagnostics)
        / sum(x["pixels"] for x in diagnostics),
        "observed_points": len(cloud.points),
        "fusion_seconds": time.monotonic() - started,
        "loop_closure": False,
        "pose_source": inference.get("pose_source", "Lingbot sequential Sim3 windows"),
        "absolute_dimensions_verified": False,
        "unseen_surfaces_completed": False,
        "limitations": [
            "Monocular scale lacks measured anchors",
            "Sequential window alignment can drift",
            "Glass, mirrors, motion and unobserved surfaces remain uncertain",
            "Depth consistency is an internal check, not independent ground truth",
        ],
        "files": {
            p.name: digest(p)
            for p in artifact.glob("*")
            if p.suffix in (".glb", ".ply")
        },
    }
    write_json(artifact / "validation.json", report)
