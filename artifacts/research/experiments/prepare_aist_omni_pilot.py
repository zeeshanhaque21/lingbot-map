"""Prepare a bounded AIST panorama experiment with calibrated perspective inputs."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from artifacts.research.experiments.compare_panorama_sources import sample_panorama


def projection(width, height, horizontal_fov, yaw, panorama_shape):
    """Return integer-centered intrinsics, camera-to-panorama rotation, and maps."""
    focal = width / (2 * math.tan(math.radians(horizontal_fov) / 2))
    intrinsic = np.array(
        [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]]
    )
    angle = math.radians(yaw)
    rotation = np.array(
        [[math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
         [-math.sin(angle), 0, math.cos(angle)]]
    )
    yy, xx = np.mgrid[:height, :width]
    rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(intrinsic).T
    rays = rays @ rotation.T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    pano_height, pano_width = panorama_shape[:2]
    map_x = (np.arctan2(rays[..., 0], rays[..., 2]) / (2 * np.pi) + 0.5) * pano_width - 0.5
    map_y = (np.arcsin(np.clip(rays[..., 1], -1, 1)) / np.pi + 0.5) * pano_height - 0.5
    return intrinsic, rotation, map_x, map_y


def aist_mask(shape):
    """Use the invalid rectangles supplied in AIST's camera configuration."""
    height, width = shape[:2]
    yy, xx = np.mgrid[:height, :width]
    x, y = (xx + 0.5) / width, (yy + 0.5) / height
    invalid = (y < 0.1) | (y >= 0.84) | ((y >= 0.7) & ((x < 0.2) | (x >= 0.8)))
    return (~invalid).astype(np.float32)


def save_image(path, image):
    if path.exists():
        raise ValueError(f"Refusing to replace {path}")
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")


def prepare(video, output, start=45.0, seconds=12.0, sample_fps=2.0):
    video, output = Path(video).resolve(), Path(output).resolve()
    if not video.is_file():
        raise ValueError("Input video does not exist")
    if output.exists():
        raise ValueError("Choose a new output directory; existing experiments are preserved")
    if start < 0 or seconds <= 0 or sample_fps <= 0 or seconds * sample_fps > 32:
        raise ValueError("Require a positive sample of at most 32 capture times")
    capture = cv2.VideoCapture(str(video))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or source_fps <= 0:
        raise ValueError("Cannot decode input video")
    count = math.ceil(seconds * sample_fps)
    indices = [round((start + i / sample_fps) * source_fps) for i in range(count)]
    if indices[-1] >= total_frames or len(set(indices)) != count:
        capture.release()
        raise ValueError("Requested sample exceeds the video or repeats source frames")
    output.mkdir(parents=True)
    for directory in ["panoramas", "single/images", "omni/images"]:
        (output / directory).mkdir(parents=True)
    frames = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            valid, frame = capture.read()
            if not valid or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != index + 1:
                raise RuntimeError(f"Could not decode exact source frame {index}")
            if frame.shape[1] != frame.shape[0] * 2:
                raise ValueError("Expected a 2:1 equirectangular video")
            frames.append(frame)
            save_image(output / "panoramas" / f"{index:06d}.png", frame)
    finally:
        capture.release()
    mask = aist_mask(frames[0].shape)
    save_image(output / "panoramas" / "source-mask.png", (mask * 255).astype(np.uint8))
    metadata = {
        "schema": 1,
        "source": str(video),
        "source_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "source_fps": source_fps,
        "source_frames": indices,
        "source_timestamps_seconds": [index / source_fps for index in indices],
        "source_projection": "equirectangular, longitude rightward, latitude downward",
        "source_mask": "panoramas/source-mask.png",
        "pixel_center_offset": 0.0,
        "perspective_size": [518, 294],
        "horizontal_fov_degrees": 100.0,
        "ordering": "direction-major, alternating temporal traversal; all views at a capture time share one optical center",
        "inference_calibration": "Known calibration is recorded for audit; native LingBot predicts its own calibration",
        "arms": {},
    }
    for arm, yaws in [("single", [0]), ("omni", [0, 90, 180, 270])]:
        rows = []
        for direction, yaw in enumerate(yaws):
            intrinsic, rotation, map_x, map_y = projection(518, 294, 100, yaw, frames[0].shape)
            valid_mask = sample_panorama(mask, map_x, map_y) > 0.999
            if not valid_mask.all():
                raise ValueError("Projection includes invalid AIST pixels; choose a supported field of view before inference")
            order = range(count) if direction % 2 == 0 else range(count - 1, -1, -1)
            for time_id in order:
                image_id = len(rows)
                image = sample_panorama(frames[time_id], map_x, map_y)
                file = f"images/{image_id:06d}.png"
                save_image(output / arm / file, image)
                rows.append({
                    "frame": image_id,
                    "file": file,
                    "capture_id": time_id,
                    "source_frame": indices[time_id],
                    "timestamp_seconds": indices[time_id] / source_fps,
                    "yaw_degrees": yaw,
                    "camera_to_panorama_rotation": rotation.tolist(),
                    "intrinsics": intrinsic.tolist(),
                    "valid_pixel_fraction": float(valid_mask.mean()),
                    "image_sha256": hashlib.sha256((output / arm / file).read_bytes()).hexdigest(),
                })
        metadata["arms"][arm] = rows
    with (output / "projection.json").open("x") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start", type=float, default=45.0)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()
    if args.video is None or args.output is None:
        print("description: Prepare one bounded AIST omnidirectional pilot")
        print('help: "Provide --video <video.mp4> --output <new-directory>; optional --start 45 --seconds 12 --fps 2"')
        return 0 if args.video is None and args.output is None else 2
    try:
        result = prepare(args.video, args.output, args.start, args.seconds, args.fps)
    except (ValueError, RuntimeError, OSError) as error:
        print("error: " + json.dumps(str(error)))
        print('help: "Check the video and sample bounds; use a new --output directory"')
        return 1
    print("status: complete")
    print("captures: " + str(len(result["source_frames"])))
    print("single_views: " + str(len(result["arms"]["single"])))
    print("omni_views: " + str(len(result["arms"]["omni"])))
    print("output: " + json.dumps(str(args.output.resolve())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
