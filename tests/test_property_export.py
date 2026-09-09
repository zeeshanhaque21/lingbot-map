import json
import struct

import numpy as np
import trimesh

from artifacts.research.experiments.texture_bake import rasterize_positions
from lingbot_map.reconstruction.materials import unlit_materials
from lingbot_map.reconstruction.repair import disconnected_joins


def test_uv_rasterization_preserves_surface_coordinates():
    vertices = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [0.0, 1, 0]])
    uv = vertices[:, :2]
    positions, occupied = rasterize_positions(
        vertices, np.array([[0, 1, 2], [0, 2, 3]]), uv, 8, 8
    )
    assert occupied.all()
    np.testing.assert_allclose(positions[0, 0], [0.0625, 0.9375, 0])
    np.testing.assert_allclose(positions[-1, -1], [0.9375, 0.0625, 0])


def test_export_preserves_vertex_colors_and_marks_captured_appearance_unlit():
    mesh = trimesh.creation.box()
    mesh.visual.vertex_colors = [120, 60, 20, 255]
    blob = mesh.export(file_type="glb", tree_postprocessor=unlit_materials)
    size = struct.unpack_from("<I", blob, 12)[0]
    tree = json.loads(blob[20 : 20 + size])
    assert "KHR_materials_unlit" in tree["extensionsUsed"]
    assert "COLOR_0" in tree["meshes"][0]["primitives"][0]["attributes"]
    material = tree["meshes"][0]["primitives"][0]["material"]
    assert tree["materials"][material]["extensions"]["KHR_materials_unlit"] == {}


def test_repair_connects_camera_groups_across_an_isolated_bad_frame():
    joins = disconnected_joins(
        {"edge_frames": [[0, 1], [1, 2], [4, 5], [5, 6], [7, 8], [8, 9]]}, 10
    )
    assert joins == [(2, 4), (6, 7)]
