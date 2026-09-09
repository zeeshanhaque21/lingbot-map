"""Explicit checkpoint pose conventions at the inference boundary."""

import numpy as np

LONG_CHECKPOINT_SHA256 = (
    "832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409"
)


def world_to_camera(decoded, convention):
    """Normalize decoded [R|t] matrices to the pipeline's world-to-camera format."""
    if convention == "world-to-camera":
        return decoded.copy()
    if convention != "camera-to-world":
        raise ValueError("Specify camera-to-world or world-to-camera checkpoint poses")
    result = decoded.copy()
    result[..., :3, :3] = np.swapaxes(decoded[..., :3, :3], -1, -2)
    result[..., :3, 3] = -np.einsum(
        "...ji,...j->...i", decoded[..., :3, :3], decoded[..., :3, 3]
    )
    return result
