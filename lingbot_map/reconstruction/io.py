"""Durable artifacts and deterministic video sampling."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def prepare(video, output, fps=2.0, limit=None):
    video, output = Path(video).resolve(), Path(output)
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
        "-show_format", "-of", "json", str(video),
    ]))
    signature = {"source": str(video), "source_sha256": digest(video),
                 "fps": fps, "limit": limit, "width": 1280, "schema": 1}
    manifest_path = output / "input.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["configuration"] != signature:
            raise ValueError("Output belongs to another input/configuration; choose a new --output")
        if all((output / frame["file"]).is_file() for frame in manifest["frames"]):
            return manifest
        raise ValueError("Input manifest has missing frames; choose a new --output")
    frames = output / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    # A partial extraction can be resumed without replacing already saved frames.
    existing = sorted(frames.glob("*.jpg"))
    start = len(existing)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n"]
    if sys.platform == "darwin":
        command += ["-hwaccel", "videotoolbox"]
    command += ["-i", str(video), "-vf",
               f"fps={fps}:start_time=0,select='gte(n,{start})',scale=1280:-2",
               "-fps_mode", "vfr", "-start_number", str(start), "-q:v", "2"]
    if limit is not None:
        if start < limit:
            command += ["-frames:v", str(limit - start)]
        else:
            command = None
    if command:
        subprocess.run(command + [str(frames / "%06d.jpg")], check=True)
    files = sorted(frames.glob("*.jpg"))
    if not files:
        raise ValueError("Video produced no frames")
    manifest = {"configuration": signature, "probe": probe,
                "coordinate_convention": "OpenCV camera: right, down, forward; world-to-camera extrinsics",
                "units": "uncalibrated model units",
                "frames": [{"id": i, "timestamp_seconds": i / fps,
                            "file": str(f.relative_to(output))} for i, f in enumerate(files)]}
    write_json(manifest_path, manifest)
    return manifest
