"""Import ordered photographs without resampling pixels or inventing capture time."""

import json
import re
import shutil
from pathlib import Path

from PIL import Image

from .io import digest, write_json


def frame_label(camera):
    label = f"Frame {camera['frame']}"
    seconds = camera.get("timestamp_seconds")
    return label if seconds is None else f"{label} · {seconds:.1f}s"


def prepare_images(source, output, limit=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError("--images must identify one ordered image directory")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    files = sorted(
        (
            p
            for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ),
        key=lambda p: [
            int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.name)
        ],
    )
    if not files:
        raise ValueError(
            "No PNG or JPEG images found; select one sequence, not its parent"
        )
    files = files[:limit]
    rows = []
    for i, path in enumerate(files):
        with Image.open(path) as image:
            if image.getexif().get(274, 1) != 1:
                raise ValueError(
                    f"Normalize EXIF orientation explicitly before import: {path}"
                )
            if image.mode != "RGB":
                raise ValueError(
                    f"Expected RGB photographs without transparency: {path}"
                )
            image.load()
            size = list(image.size)
        rows.append(
            {
                "id": i,
                "timestamp_seconds": None,
                "file": f"frames/{i:06d}{path.suffix.lower()}",
                "source_file": path.name,
                "source_sha256": digest(path),
                "size": size,
            }
        )
    if len({tuple(row["size"]) for row in rows}) != 1:
        raise ValueError("Images must share dimensions for the current camera model")
    manifest = {
        "configuration": {
            "source": str(source),
            "source_kind": "image_sequence",
            "fps": None,
            "limit": limit,
            "width": rows[0]["size"][0],
            "schema": 1,
        },
        "coordinate_convention": "OpenCV camera: right, down, forward; world-to-camera extrinsics",
        "units": "uncalibrated model units",
        "ordering": "Natural filename order; capture times were not supplied",
        "frames": rows,
    }
    manifest_path = output / "input.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(
            "Image sequence changed; preserve this run and choose a new --output"
        )
    (output / "frames").mkdir(parents=True, exist_ok=True)
    expected = {row["file"] for row in rows}
    if any(
        str(p.relative_to(output)) not in expected
        for p in (output / "frames").iterdir()
    ):
        raise ValueError("Output contains unrelated frames; choose a new --output")
    for row, path in zip(rows, files, strict=True):
        target = output / row["file"]
        if target.exists():
            if digest(target) != row["source_sha256"]:
                raise ValueError(f"Saved frame differs from its source: {target}")
        else:
            shutil.copy2(path, target)
            if digest(target) != row["source_sha256"]:
                raise ValueError(f"Image changed during import: {path}")
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    return manifest
