"""Clean a navigation mesh and render fixed spherical property-tour stations."""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

from .closure import audit_closure
from .io import digest, write_json
from .materials import unlit_materials
from .rendering import SurfaceRenderer


def panorama_rays(pose, width, height):
    """Equirectangular centers: image center looks forward, image top looks up."""
    pose = np.asarray(pose, dtype=float)
    if width < 1 or height < 1:
        raise ValueError("Panorama dimensions must be positive")
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1])
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)
        or not np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-5)
    ):
        raise ValueError("Panorama station requires a rigid camera-to-world pose")
    yaw = ((np.arange(width) + 0.5) / width - 0.5) * 2 * np.pi
    pitch = (0.5 - (np.arange(height) + 0.5) / height) * np.pi
    y, p = np.meshgrid(yaw, pitch)
    directions = np.stack(
        [np.cos(p) * np.sin(y), -np.sin(p), np.cos(p) * np.cos(y)], axis=-1
    )
    rays = np.empty((height, width, 6), dtype=np.float32)
    rays[..., :3] = pose[:3, 3]
    rays[..., 3:] = directions @ pose[:3, :3].T
    return rays


def select_stations(cameras, count):
    if not cameras or count < 1:
        raise ValueError("Station selection requires cameras and a positive count")
    positions = np.array([camera["camera_to_world"] for camera in cameras])[:, :3, 3]
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    if distance[-1] <= 0 or count < 1:
        raise ValueError(
            "Station selection requires a moving capture and a positive count"
        )
    selected = []
    spacing = distance[-1] / max(count, 1)
    for target in np.linspace(0, distance[-1], count, endpoint=False):
        index = int(np.argmin(np.abs(distance - target)))
        if (
            selected
            and min(np.linalg.norm(positions[index] - positions[j]) for j in selected)
            < spacing * 0.2
        ):
            continue
        selected.append(index)
    return selected


def captured_rgb(source, camera):
    if camera.get("captured_image"):
        return np.asarray(Image.open(camera["captured_image"]).convert("RGB"))
    windows = sorted((source / "windows").glob("*.npz"))
    with np.load(windows[camera["window"]]) as data:
        index = int(np.flatnonzero(data["frame_ids"] == camera["frame"])[0])
        return data["rgb"][index].copy()


def export_tour(
    source, output, station_count=6, panorama_width=2048, triangle_budget=150000
):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Preserve existing tours; choose a new output directory")
    if panorama_width < 64 or panorama_width % 2 or triangle_budget < 1000:
        raise ValueError("Use an even panorama width >=64 and a triangle budget >=1000")
    cameras = json.loads((source / "model/cameras.json").read_text())
    chosen = select_stations(cameras, station_count)
    mesh_path = source / "model/observed-surfaces.ply"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles() or not mesh.has_vertex_colors():
        raise ValueError("Tour export requires a colored observed surface mesh")
    original_triangles = len(mesh.triangles)
    clusters, counts, areas = mesh.cluster_connected_triangles()
    clusters, counts, areas = (
        np.asarray(clusters),
        np.asarray(counts),
        np.asarray(areas),
    )
    tiny = (counts < 64) & (areas < areas.sum() * 1e-5)
    mesh.remove_triangles_by_mask(tiny[clusters])
    mesh.remove_unreferenced_vertices()
    cleaned_triangles = len(mesh.triangles)
    output.mkdir(parents=True)
    o3d.io.write_triangle_mesh(str(output / "render-mesh.ply"), mesh)
    renderer = SurfaceRenderer(mesh)
    navigation = (
        mesh.simplify_quadric_decimation(triangle_budget)
        if len(mesh.triangles) > triangle_budget
        else mesh
    )
    vertices = np.asarray(navigation.vertices).copy()
    vertices[:, 1:] *= -1
    asset = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(navigation.triangles),
        vertex_colors=(np.clip(np.asarray(navigation.vertex_colors), 0, 1) * 255)
        .round()
        .astype(np.uint8),
        process=False,
    )
    asset.export(str(output / "property.glb"), tree_postprocessor=unlit_materials)
    recovered = trimesh.load(
        output / "property.glb", force="mesh", process=False, skip_materials=True
    )
    if not np.array_equal(asset.faces, recovered.faces) or not np.allclose(
        asset.vertices, recovered.vertices, atol=1e-6
    ):
        raise ValueError("Navigation export changed the simplified mesh")
    stations = []
    for n, index in enumerate(chosen):
        camera = cameras[index]
        pose = np.asarray(camera["camera_to_world"])
        root = output / "stations" / f"{n:03d}"
        root.mkdir(parents=True)
        rgb, radial_depth, visible = renderer.render_rays(
            panorama_rays(pose, panorama_width, panorama_width // 2)
        )
        Image.fromarray(rgb).save(root / "panorama.png")
        Image.fromarray(visible.astype(np.uint8) * 255).save(root / "observed-mask.png")
        np.save(root / "radial-depth.npy", np.where(visible, radial_depth, 0))
        reference = captured_rgb(source, camera)
        Image.fromarray(reference).save(root / "reference.png")
        stations.append(
            {
                "id": f"{n:03d}",
                "frame": camera["frame"],
                "camera_to_world": pose.tolist(),
                "intrinsics": camera["intrinsics"],
                "reference_size": [reference.shape[1], reference.shape[0]],
                "panorama": f"stations/{n:03d}/panorama.png",
                "observed_mask": f"stations/{n:03d}/observed-mask.png",
                "observed_sphere_fraction": float(visible.mean()),
                "appearance": "raw_mesh_render",
                "generated": False,
                "panorama_sha256": digest(root / "panorama.png"),
            }
        )
        print(
            json.dumps(
                {
                    "station": n,
                    "frame": camera["frame"],
                    "observed_sphere_fraction": float(visible.mean()),
                }
            ),
            flush=True,
        )
        write_json(output / "stations.json", stations)
    edges = [
        [stations[i]["id"], stations[i + 1]["id"]] for i in range(len(stations) - 1)
    ]
    closure = audit_closure(source, cameras)
    write_json(output / "closure-audit.json", closure)
    if closure["navigation_closure"] and len(stations) > 2:
        edges.append([stations[-1]["id"], stations[0]["id"]])
    result = {
        "source": str(source),
        "source_mesh_sha256": digest(mesh_path),
        "source_cameras_sha256": digest(source / "model/cameras.json"),
        "source_triangles": original_triangles,
        "cleaned_triangles": cleaned_triangles,
        "removed_tiny_components": int(tiny.sum()),
        "navigation_triangles": len(asset.faces),
        "navigation_bytes": (output / "property.glb").stat().st_size,
        "navigation_sha256": digest(output / "property.glb"),
        "station_count": len(stations),
        "navigation_edges": edges,
        "loop_constraints_verified": closure["navigation_closure"],
        "closure_audit": "closure-audit.json",
        "world_convention": "OpenCV world, camera-to-world station poses; app GLB flips world Y/Z",
        "panorama_convention": "Equirectangular 2:1, center +Z, right +X, top -Y in station camera coordinates",
        "metric_accuracy_verified": False,
        "fixanything_refinement": "not_run",
    }
    write_json(output / "tour.json", result)
    return result
