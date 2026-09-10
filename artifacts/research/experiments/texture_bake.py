"""Bake observed mesh colors onto the simplified app asset without generating pixels."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

from lingbot_map.reconstruction.io import digest, write_json
from lingbot_map.reconstruction.materials import unlit_materials


def rasterize_positions(vertices, triangles, uv, width, height):
    """Rasterize UV triangle positions at texture pixel centers."""
    positions = np.zeros((height, width, 3), np.float32)
    occupied = np.zeros((height, width), bool)
    coordinates = np.column_stack(
        [uv[:, 0] * width - 0.5, (1 - uv[:, 1]) * height - 0.5]
    )
    for triangle in triangles:
        a, b, c = coordinates[triangle]
        low = np.maximum(np.floor(np.minimum(np.minimum(a, b), c)).astype(int), 0)
        high = np.minimum(
            np.ceil(np.maximum(np.maximum(a, b), c)).astype(int),
            [width - 1, height - 1],
        )
        if np.any(high < low):
            continue
        y, x = np.mgrid[low[1] : high[1] + 1, low[0] : high[0] + 1]
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-10:
            continue
        wa = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
        wb = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
        wc = 1 - wa - wb
        inside = (wa >= -1e-6) & (wb >= -1e-6) & (wc >= -1e-6)
        points = (
            wa[..., None] * vertices[triangle[0]]
            + wb[..., None] * vertices[triangle[1]]
            + wc[..., None] * vertices[triangle[2]]
        )
        positions[y[inside], x[inside]] = points[inside]
        occupied[y[inside], x[inside]] = True
    return positions, occupied


def texture(output, resolution=4096):
    import xatlas

    root = Path(output) / "model"
    signature = {
        "source_mesh": digest(root / "observed-surfaces.ply"),
        "simplified_mesh": digest(root / "property.glb"),
        "resolution": resolution,
        "schema": 2,
        "target_charts": 2048,
    }
    marker = root / "texture.json"
    target = root / "property-textured.glb"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if (
            previous["input"] == signature
            and target.exists()
            and digest(target) == previous["output_sha256"]
        ):
            return
        raise ValueError(
            "Texture inputs or output changed; preserve artifacts and use a new output directory"
        )
    full = o3d.io.read_triangle_mesh(str(root / "observed-surfaces.ply"))
    simplified = trimesh.load(root / "property.glb", force="mesh", process=False)
    simplified.apply_transform(np.diag([1.0, -1.0, -1.0, 1.0]))
    atlas = xatlas.Atlas()
    atlas.add_mesh(
        np.asarray(simplified.vertices, np.float32),
        np.asarray(simplified.faces, np.uint32),
    )
    options = xatlas.PackOptions()
    options.resolution, options.padding = resolution, 4
    charts = xatlas.ChartOptions()
    charts.max_chart_area = float(simplified.area / 2048)
    charts.max_boundary_length = float(4 * np.sqrt(charts.max_chart_area))
    print("Texture: building bounded UV charts", file=sys.stderr, flush=True)
    atlas.generate(chart_options=charts, pack_options=options)
    print("Texture: rasterizing observed surface colors", file=sys.stderr, flush=True)
    mapping, faces, uv = atlas[0]
    vertices = simplified.vertices[mapping]
    positions, occupied = rasterize_positions(
        vertices, faces, uv, atlas.width, atlas.height
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(full))
    colors, full_faces = np.asarray(full.vertex_colors), np.asarray(full.triangles)
    points = positions[occupied]
    baked = np.zeros((len(points), 3), np.uint8)
    distances = []
    for start in range(0, len(points), 100000):
        query = points[start : start + 100000]
        closest = scene.compute_closest_points(
            o3d.core.Tensor(query, o3d.core.Dtype.Float32)
        )
        primitive = closest["primitive_ids"].numpy()
        bary = closest["primitive_uvs"].numpy()
        weights = np.column_stack([1 - bary.sum(1), bary])
        baked[start : start + len(query)] = (
            np.clip(
                255
                * np.sum(colors[full_faces[primitive]] * weights[..., None], axis=1),
                0,
                255,
            )
            .round()
            .astype(np.uint8)
        )
        distances.append(
            np.linalg.norm(closest["points"].numpy() - query, axis=1)[::64]
        )
    pixels = np.zeros((atlas.height, atlas.width, 3), np.uint8)
    pixels[occupied] = baked
    # Pad chart borders for bilinear sampling; this changes texture padding only.
    for _ in range(4):
        expanded = cv2.dilate(
            occupied.astype(np.uint8), np.ones((3, 3), np.uint8)
        ).astype(bool)
        border = expanded & ~occupied
        values = cv2.blur(pixels.astype(np.float32), (3, 3))
        weights = cv2.blur(occupied.astype(np.float32), (3, 3))
        pixels[border] = (
            np.clip(values[border] / np.maximum(weights[border, None], 1e-8), 0, 255)
            .round()
            .astype(np.uint8)
        )
        occupied = expanded
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(pixels),
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True,
    )
    vertices = vertices.copy()
    vertices[:, 1:] *= -1
    asset = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=trimesh.visual.texture.TextureVisuals(uv=uv, material=material),
        process=False,
    )
    asset.metadata.update(
        units="uncalibrated model units",
        evidence="Colors baked from the observed mesh; no generated texture",
    )
    asset.export(target, tree_postprocessor=unlit_materials)
    write_json(
        marker,
        {
            "input": signature,
            "output_sha256": digest(target),
            "texture_dimensions": [atlas.width, atlas.height],
            "observed_texels": len(points),
            "surface_distance_p50_p95_p99": np.quantile(
                np.concatenate(distances), [0.5, 0.95, 0.99]
            ).tolist(),
            "geometry_source": "property.glb",
            "color_source": "observed-surfaces.ply",
            "generated_content": False,
        },
    )
