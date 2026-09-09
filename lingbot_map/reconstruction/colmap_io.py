"""Read COLMAP's documented text model without depending on pycolmap."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def read_model(path):
    path = Path(path)
    cameras, images, points = {}, {}, {}
    for line in (path / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if fields[1] != "PINHOLE":
            raise ValueError(
                "Camera refinement requires PINHOLE images; undistort other camera models first"
            )
        fx, fy, cx, cy = map(float, fields[4:])
        cameras[int(fields[0])] = {
            "width": int(fields[2]),
            "height": int(fields[3]),
            "intrinsics": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]),
        }
    lines = iter((path / "images.txt").read_text().splitlines())
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        quaternion = np.asarray(fields[1:5], float)
        rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        extrinsic = np.column_stack([rotation, np.asarray(fields[5:8], float)])
        observations = next(lines).split()
        observations = np.asarray(observations, float).reshape(-1, 3)
        images[fields[9]] = {
            "id": int(fields[0]),
            "camera": cameras[int(fields[8])],
            "extrinsics": extrinsic,
            "observations": observations,
        }
    for line in (path / "points3D.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        points[int(fields[0])] = {
            "xyz": np.asarray(fields[1:4], float),
            "error": float(fields[7]),
            "track_length": len(fields[8:]) // 2,
        }
    return images, points


def captured_image(images, frame):
    """Resolve a numbered capture without silently choosing between duplicate names."""
    names = [f"{int(frame):06d}{suffix}" for suffix in (".jpg", ".jpeg", ".png")]
    matches = [images[name] for name in names if name in images]
    if len(matches) != 1:
        raise ValueError(f"Expected one calibrated photograph for frame {frame}; found {len(matches)}")
    return matches[0]
