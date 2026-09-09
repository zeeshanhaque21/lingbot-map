"""Fuse saved native stereo depths while requiring another view to support each sample."""

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

from artifacts.research.experiments.openmvs_depth import read_depth_map
from lingbot_map.reconstruction.colmap_io import read_model
from lingbot_map.reconstruction.geometry import supported_depth
from lingbot_map.reconstruction.io import digest, write_json
from lingbot_map.reconstruction.materials import unlit_materials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trial", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--colmap-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-divisor", type=float, default=360)
    parser.add_argument("--frame-range", type=int, nargs=2, metavar=("FIRST", "LAST"))
    args = parser.parse_args()
    if args.output.exists() or args.depth_divisor <= 0:
        parser.error("Use a fresh output directory and a positive depth divisor")
    source, dataset, output = (
        args.source_trial.resolve(),
        args.dataset.resolve(),
        args.output.resolve(),
    )
    source_run = json.loads((source / "run.json").read_text())
    provenance = json.loads((dataset / "provenance.json").read_text())
    if digest(dataset / "provenance.json") != source_run["dataset_provenance_sha256"]:
        raise ValueError("Depth maps and prepared dataset have different provenance")
    if any(digest(args.colmap_text / name) != sha for name, sha in provenance["source_colmap_sha256"].items()):
        raise ValueError("Calibration changed after native dataset preparation")
    images, _ = read_model(args.colmap_text)
    metadata = [
        read_depth_map(path, metadata_only=True)
        for path in sorted(source.glob("*.dmap"))
    ]
    if not metadata:
        raise ValueError("No saved native depth maps")
    lookup = {int(item["ids"][0]): item for item in metadata}
    if len(lookup) != len(metadata):
        raise ValueError("Duplicate native depth view IDs")
    if args.frame_range:
        first, last = args.frame_range
        if first > last:
            parser.error("FIRST must not exceed LAST")
        selected = [item for item in metadata if first <= int(item["image"].stem) <= last]
    else:
        selected = metadata
    if not selected:
        raise ValueError("No native depth maps in the selected range")
    hashes = {}
    for item in metadata:
        image = item["image"]
        relative = str(image.relative_to(dataset / "images"))
        if digest(image) != provenance["image_sha256"][relative]:
            raise ValueError(
                f"Native depth map uses an unverified or reserved image: {image}"
            )
        calibration = (
            images[image.with_suffix(".jpg").name]
            if image.name not in images
            else images[image.name]
        )
        k = calibration["camera"]["intrinsics"].copy()
        k[:2, 2] -= 0.5
        if item["image_size"] != item["depth_size"] or not np.allclose(
            item["intrinsics"], k, atol=1e-7, rtol=0
        ):
            raise ValueError(
                "Native depth pixel convention differs from the calibrated capture"
            )
        if not np.allclose(
            item["extrinsics"][:3], calibration["extrinsics"], atol=1e-7, rtol=0
        ):
            raise ValueError("Native depth pose differs from the calibrated capture")
        hashes[item["path"].name] = digest(item["path"])
    depth_medians = []
    for i in np.linspace(0, len(selected) - 1, min(31, len(selected)), dtype=int):
        depth = read_depth_map(selected[i]["path"])["depth"]
        valid = np.isfinite(depth) & (depth > 0)
        if valid.any():
            depth_medians.append(float(np.median(depth[valid])))
    voxel = float(np.median(depth_medians) / args.depth_divisor)
    if not np.isfinite(voxel) or voxel <= 0:
        raise ValueError("Native maps have no valid depth scale")
    output.mkdir(parents=True)
    write_json(
        output / "run.json",
        {
            **source_run,
            "source_trial": str(source),
            "method": "native_stereo_tsdf",
            "native_depth_sha256": hashes,
            "voxel_size_model_units": voxel,
            "depth_divisor": args.depth_divisor,
            "frame_range": args.frame_range,
            "integration_frames": [int(item["image"].stem) for item in selected],
            "depth_scale_samples": depth_medians,
            "producer_sources_sha256": {
                str(path): digest(path)
                for path in [
                    Path(__file__),
                    Path(read_depth_map.__code__.co_filename),
                    Path(supported_depth.__code__.co_filename),
                ]
            },
            "interpretation": "Native stereo depths, unchanged calibrated cameras, 0.5-pixel COLMAP-to-native principal-point shift verified. Samples require another native view within 3% depth and pass confidence/edge filtering. No watertight completion. Monocular scale and layout remain unverified.",
        },
    )
    (output / "dense.ply").symlink_to(source / "dense.ply")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=voxel * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    @lru_cache(maxsize=9)
    def load(identifier):
        item = read_depth_map(lookup[identifier]["path"])
        if digest(item["path"]) != hashes[item["path"].name]:
            raise ValueError("Native depth changed during fusion")
        return item

    began, integrated = time.monotonic(), 0
    with (output / "frames.jsonl").open("x") as trace:
        for meta in selected:
            ref = load(int(meta["ids"][0]))
            neighbors = [load(int(i)) for i in meta["ids"][1:] if int(i) in lookup][:2]
            if not neighbors:
                continue
            samples = (
                [ref, neighbors[0]]
                if len(neighbors) == 1
                else [ref, neighbors[0], ref, neighbors[1]]
            )
            data = {
                key: [item[key] for item in samples]
                for key in ["depth", "confidence", "intrinsics", "extrinsics"]
            }
            data["extrinsics"] = [item["extrinsics"][:3] for item in samples]
            data["frame_ids"] = [int(item["image"].stem) for item in samples]
            filtered, stats = supported_depth(0, data)
            rgb = np.asarray(Image.open(ref["image"]).convert("RGB"))
            w, h = ref["depth_size"]
            k = ref["intrinsics"]
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb.copy()),
                o3d.geometry.Image(filtered),
                depth_scale=1,
                depth_trunc=float(filtered.max() + 1),
                convert_rgb_to_intensity=False,
            )
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    w, h, k[0, 0], k[1, 1], k[0, 2], k[1, 2]
                ),
                ref["extrinsics"],
            )
            integrated += 1
            trace.write(json.dumps(stats) + "\n")
            trace.flush()
            if integrated % 25 == 0:
                print(
                    json.dumps(
                        {
                            "integrated": integrated,
                            "available": len(selected),
                            "seconds": time.monotonic() - began,
                        }
                    ),
                    flush=True,
                )
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
    if not mesh.has_triangles():
        raise ValueError("No native stereo surface survived fusion")
    o3d.io.write_triangle_mesh(str(output / "mesh.ply"), mesh)
    asset = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=(np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255)
        .round()
        .astype(np.uint8),
        process=False,
    )
    asset.export(str(output / "property.glb"), tree_postprocessor=unlit_materials)
    result = {
        "complete": True,
        "property_sha256": digest(output / "property.glb"),
        "triangles": len(asset.faces),
        "integrated_images": integrated,
        "seconds": time.monotonic() - began,
        "voxel_size_model_units": voxel,
        "metric_accuracy_verified": False,
    }
    write_json(output / "completion.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
