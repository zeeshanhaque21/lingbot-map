"""Prepare one complete source-assisted sweep without running generation."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from lingbot_map.reconstruction.io import digest, write_json
from lingbot_map.reconstruction.rendering import SurfaceRenderer
from lingbot_map.reconstruction.source_photos import load_photos, project_photos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-photos", type=int, default=24)
    args = parser.parse_args()
    config = json.loads((args.sweep / "sweep.json").read_text())
    tour = Path(config["tour"])
    tour_config = json.loads((tour / "tour.json").read_text())
    source = Path(tour_config["source"])
    renderer = SurfaceRenderer.from_file(tour / "render-mesh.ply")
    photos = load_photos(source, maximum=args.maximum_photos)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ["frames", "raw-frames", "source-frame-maps"]:
        (args.output / name).mkdir()
    height, width = config["height"], config["width"]
    y, x = np.indices((height, width))
    camera_rays = (
        np.stack([x, y, np.ones_like(x)], -1) @ np.linalg.inv(config["intrinsics"]).T
    )
    k = np.asarray(config["intrinsics"])
    k[:2, 2] += 0.5
    for frame in config["frames"]:
        index = frame["index"]
        pose = np.asarray(frame["camera_to_world"])
        rgb, depth, visible = renderer.render(k, np.linalg.inv(pose), width, height)
        # Mesh depth is camera Z, not Euclidean distance.
        world = (camera_rays * np.where(visible, depth, 0)[..., None]) @ pose[
            :3, :3
        ].T + pose[:3, 3]
        refined, provenance = project_photos(world, visible, rgb, pose[:3, 3], photos)
        name = f"{index:03d}.png"
        Image.fromarray(rgb).save(args.output / "raw-frames" / name)
        if frame["captured_anchor"]:
            shutil.copyfile(args.sweep / "frames" / name, args.output / "frames" / name)
            provenance.fill(-1)
        else:
            Image.fromarray(refined).save(args.output / "frames" / name)
        np.save(args.output / "source-frame-maps" / f"{index:03d}.npy", provenance)
        ids, counts = np.unique(provenance[provenance >= 0], return_counts=True)
        frame.update(
            source_photo_fraction=float(np.mean(provenance >= 0)),
            source_photo_pixels={str(i): int(c) for i, c in zip(ids, counts)},
            observed_fraction=float(visible.mean()),
        )
        write_json(args.output / "frames.json", config["frames"])
        print(
            json.dumps(
                {
                    "frame": index,
                    "photo_fraction": frame["source_photo_fraction"],
                    "photos": len(ids),
                }
            ),
            flush=True,
        )
    config.update(
        source_sweep_sha256=digest(args.sweep / "sweep.json"),
        conditioning="captured RGB projected onto depth-consistent visible mesh surfaces",
        source_photos=[
            {k: p[k] for k in ["frame", "path", "sha256", "depth_archive_sha256"]}
            for p in photos
        ],
        source_photo_tolerance=0.03,
        reserved_source_frames_excluded=True,
        panorama_quality_verified=False,
    )
    write_json(args.output / "sweep.json", config)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "frames": len(config["frames"]),
                "source_photos": len(photos),
                "generation_started": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
