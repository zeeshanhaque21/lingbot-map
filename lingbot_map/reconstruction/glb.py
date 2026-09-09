"""Preserve native textured GLB data while embedding images and converting axes."""

import base64
import json
import struct
from urllib.parse import unquote

import numpy as np
import trimesh

from .io import digest

AXIS = np.diag([1.0, -1.0, -1.0, 1.0])


def glb_document(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data)
    size, kind = struct.unpack_from("<II", data, 12)
    if magic != b"glTF" or version != 2 or length != len(data) or kind != 0x4E4F534A:
        raise ValueError("Invalid GLB container")
    return json.loads(data[20 : 20 + size])


def native_resources(trial):
    completion = json.loads((trial / "completion.json").read_text())
    source = trial / "property.glb"
    if not completion["complete"] or digest(source) != completion["property_sha256"]:
        raise ValueError("Native output is incomplete or changed")
    resources = {"property.glb": digest(source)}
    document = glb_document(source)
    for item in document.get("images", []) + document.get("buffers", []):
        uri = item.get("uri")
        if not uri or uri.startswith("data:"):
            continue
        path = (trial / unquote(uri)).resolve()
        if not path.is_relative_to(trial.resolve()):
            raise ValueError("Native resource escapes the trial directory")
        resources[uri] = digest(path)
    return resources


def embed_native_glb(source, destination):
    """Embed image bytes and add an axis parent without reserializing mesh materials."""
    if destination.exists():
        raise ValueError("Preserve the existing destination asset")
    document = glb_document(source)
    buffers = document.get("buffers", [])
    if len(buffers) != 1 or "uri" in buffers[0] or len(document.get("scenes", [])) != 1:
        raise ValueError("Expected one native scene and one embedded geometry buffer")
    original = source.read_bytes()
    json_size = struct.unpack_from("<I", original, 12)[0]
    bin_offset = 20 + json_size
    bin_size, kind = struct.unpack_from("<I4s", original, bin_offset)
    if kind != b"BIN\x00" or bin_offset + 8 + bin_size != len(original):
        raise ValueError("Expected one complete native binary chunk")
    if not 0 <= bin_size - buffers[0]["byteLength"] <= 3:
        raise ValueError("Native buffer length differs from its binary chunk")
    binary = bytearray(
        original[bin_offset + 8 : bin_offset + 8 + buffers[0]["byteLength"]]
    )
    for item in document.get("images", []):
        uri = item.get("uri")
        if uri is None:
            continue
        if uri.startswith("data:"):
            prefix, encoded = uri.split(",", 1)
            if not prefix.endswith(";base64"):
                raise ValueError("Expected base64 image data")
            data = base64.b64decode(encoded, validate=True)
        else:
            path = (source.parent / unquote(uri)).resolve()
            if not path.is_relative_to(source.parent.resolve()):
                raise ValueError("Image resource escapes the native artifact directory")
            data = path.read_bytes()
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif data.startswith(b"\xff\xd8"):
            mime = "image/jpeg"
        else:
            raise ValueError("Expected native PNG or JPEG texture bytes")
        binary.extend(b"\x00" * (-len(binary) % 4))
        views = document.setdefault("bufferViews", [])
        item.pop("uri")
        item["bufferView"] = len(views)
        item["mimeType"] = mime
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
        binary.extend(data)
    nodes = document.setdefault("nodes", [])
    # Trimesh reserves "world" for its external reference frame. A nested grouping
    # node with that name hides the added axis transform when the asset is read back.
    names = {node.get("name") for node in nodes}
    for index, node in enumerate(nodes):
        if node.get("name") == "world" and "mesh" not in node:
            name = f"native_world_{index}"
            while name in names:
                name += "_"
            node["name"] = name
            names.add(name)
    scene = document["scenes"][0]
    parent = len(nodes)
    nodes.append(
        {
            "name": f"native_app_axes_{parent}",
            "matrix": AXIS.T.ravel().tolist(),
            "children": scene["nodes"],
        }
    )
    scene["nodes"] = [parent]
    buffers[0]["byteLength"] = len(binary)
    encoded_json = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded_json += b" " * (-len(encoded_json) % 4)
    binary.extend(b"\x00" * (-len(binary) % 4))
    with destination.open("xb") as handle:
        handle.write(
            struct.pack("<4sII", b"glTF", 2, 28 + len(encoded_json) + len(binary))
        )
        handle.write(struct.pack("<I4s", len(encoded_json), b"JSON"))
        handle.write(encoded_json)
        handle.write(struct.pack("<I4s", len(binary), b"BIN\x00"))
        handle.write(binary)


def package_scene(source, destination):
    """Keep each geometry, material and node; prove the exported scene round trip."""
    scene = trimesh.load_scene(source, process=False)
    if not scene.graph.nodes_geometry:
        raise ValueError("Native scene has no triangle geometry")
    embed_native_glb(source, destination)
    document = glb_document(destination)
    if document.get("materials") != glb_document(source).get("materials"):
        raise ValueError("Packaging changed native material behavior")
    if any(
        "uri" in item
        for item in document.get("images", []) + document.get("buffers", [])
    ):
        raise ValueError("Export still needs external resources")
    recovered = trimesh.load_scene(destination, process=False)
    if set(scene.graph.nodes_geometry) != set(recovered.graph.nodes_geometry):
        raise ValueError("Packaging changed scene nodes")
    rows = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        after_transform, after_name = recovered.graph[node]
        before, after = scene.geometry[name], recovered.geometry[after_name]
        if not np.allclose(after_transform, AXIS @ transform, atol=1e-7, rtol=0):
            raise ValueError("Packaging changed node placement")
        if not np.array_equal(before.faces, after.faces) or not np.allclose(
            before.vertices, after.vertices, atol=1e-7, rtol=0
        ):
            raise ValueError("Packaging changed geometry")
        texture = getattr(before.visual.material, "baseColorTexture", None)
        after_texture = getattr(after.visual.material, "baseColorTexture", None)
        if texture is None or after_texture is None:
            raise ValueError("Expected native textured geometry")
        if not np.array_equal(np.asarray(texture), np.asarray(after_texture)):
            raise ValueError("Packaging changed texture pixels")
        if not np.allclose(before.visual.uv, after.visual.uv, atol=1e-7, rtol=0):
            raise ValueError("Packaging changed texture coordinates")
        rows.append(
            {
                "node": node,
                "triangles": len(before.faces),
                "texture_size": list(texture.size),
            }
        )
    return rows
