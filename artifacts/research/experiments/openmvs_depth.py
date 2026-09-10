"""Read the documented little-endian OpenMVS v2.4.0 raw depth-map format."""

import struct
from pathlib import Path

import numpy as np


def read_depth_map(path, metadata_only=False):
    path = Path(path)
    with path.open("rb") as stream:

        def read(size):
            value = stream.read(size)
            if len(value) != size:
                raise ValueError(f"Truncated native depth map: {path}")
            return value

        magic, flags, _, iw, ih, w, h, minimum, maximum = struct.unpack(
            "<HBBIIIIff", read(28)
        )
        if magic != 0x5244 or not flags & 1 or not 0 < w <= iw or not 0 < h <= ih:
            raise ValueError(f"Invalid native depth header: {path}")
        name = read(struct.unpack("<H", read(2))[0]).decode("utf-8")
        count = struct.unpack("<I", read(4))[0]
        if not 0 < count < 256:
            raise ValueError(f"Invalid native neighbor count: {path}")
        ids = np.frombuffer(read(count * 4), dtype="<u4").copy()
        intrinsic = np.frombuffer(read(72), dtype="<f8").reshape(3, 3).copy()
        rotation = np.frombuffer(read(72), dtype="<f8").reshape(3, 3).copy()
        center = np.frombuffer(read(24), dtype="<f8").copy()
        if (
            not np.isfinite(intrinsic).all()
            or not np.isfinite(center).all()
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
        ):
            raise ValueError(f"Invalid native calibration: {path}")
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = -rotation @ center
        expected_size = stream.tell() + w * h * (
            4 + bool(flags & 2) * 12 + bool(flags & 4) * 4 + bool(flags & 8) * 4
        )
        if path.stat().st_size != expected_size:
            raise ValueError(f"Unexpected native depth payload size: {path}")
        result = {
            "path": path,
            "image": (path.parent / name).resolve(),
            "ids": ids,
            "image_size": (iw, ih),
            "depth_size": (w, h),
            "intrinsics": intrinsic,
            "extrinsics": extrinsic,
            "minimum": minimum,
            "maximum": maximum,
        }
        if metadata_only:
            return result
        result["depth"] = (
            np.frombuffer(read(w * h * 4), dtype="<f4").reshape(h, w).copy()
        )
        if flags & 2:
            stream.seek(w * h * 12, 1)
        if not flags & 4:
            raise ValueError(f"Native confidence is required for fusion: {path}")
        result["confidence"] = (
            np.frombuffer(read(w * h * 4), dtype="<f4").reshape(h, w).copy()
        )
        return result
