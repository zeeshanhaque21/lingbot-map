import numpy as np
import pytest
import torch

from artifacts.research.experiments.mapanything_mac import (
    conditioned_views,
    local_dino_hub,
    pack_predictions,
)


def predictions():
    """Independent pinhole scene with a nonidentity camera-to-world transform."""
    k = torch.tensor([[3.0, 0, 1], [0, 4.0, 0.5], [0, 0, 1]])
    y, x = torch.meshgrid(torch.arange(2), torch.arange(3), indexing="ij")
    pixels = torch.stack([x, y, torch.ones_like(x)], -1).float()
    camera = (pixels @ torch.linalg.inv(k).T) * 2
    pose = torch.tensor([[0.0, -1, 0, 4], [1, 0, 0, 3], [0, 0, 1, 2], [0, 0, 0, 1]])
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    mask = torch.ones(1, 2, 3, dtype=torch.bool)
    mask[0, 0, 0] = False
    view = {
        "depth_z": camera[None, ..., 2:3],
        "camera_poses": pose[None],
        "intrinsics": k[None],
        "conf": torch.ones(1, 2, 3) * 2,
        "non_ambiguous_mask": mask,
        "img_no_norm": torch.ones(1, 2, 3, 3) * 0.5,
        "pts3d_cam": camera[None],
        "pts3d": world[None],
    }
    return [view, {key: value.clone() for key, value in view.items()}]


def test_raw_predictions_export_correct_camera_coordinates_and_masks():
    views = predictions()
    arrays, report = pack_predictions(views)
    world = views[0]["pts3d"][0].numpy()
    extrinsic = arrays["extrinsics"][0]
    recovered = world @ extrinsic[:, :3].T + extrinsic[:, 3]
    np.testing.assert_allclose(recovered, views[0]["pts3d_cam"][0], atol=1e-6)
    assert arrays["confidence"][0, 0, 0] == -1
    assert arrays["rgb"][0, 0, 0].tolist() == [128, 128, 128]
    assert report["pinhole_relative_error_p50_p95_p99"][-1] < 1e-6
    assert report["native_c2w_world_error_relative_p99"] < 1e-6


def test_export_rejects_wrong_camera_convention():
    views = predictions()
    views[0]["camera_poses"] = torch.linalg.inv(views[0]["camera_poses"])
    with pytest.raises(ValueError, match="C2W normalization"):
        pack_predictions(views)


def test_hub_cannot_fetch_backbone_weights(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(torch.hub, "load", lambda *a, **kw: calls.append((a, kw)))
    with local_dino_hub(tmp_path):
        with pytest.raises(ValueError, match="no automatic downloads"):
            torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14", pretrained=True)
        torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14", pretrained=False)
    assert len(calls) == 1
    assert calls[0][0][0] == str(tmp_path)
    assert calls[0][1]["source"] == "local"


def test_conditioning_keeps_image_calibration_and_pose_from_same_frame(tmp_path):
    path = tmp_path / "window.npz"
    k = np.array(
        [[[4.0, 0, 1], [0, 5.0, 1], [0, 0, 1]], [[6.0, 0, 1], [0, 7.0, 1], [0, 0, 1]]],
        dtype=np.float32,
    )
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :3], 2, axis=0)
    extrinsics[1, :, 3] = [-4, -3, -2]
    rgb = np.stack([np.zeros((2, 3, 3), np.uint8), np.full((2, 3, 3), 128, np.uint8)])
    np.savez(path, frame_ids=[760, 761], rgb=rgb, intrinsics=k, extrinsics=extrinsics)
    selected = [{"id": 761}, {"id": 760}]
    views = conditioned_views([path, path], selected, "posed")
    np.testing.assert_array_equal(views[0]["intrinsics"][0], k[1])
    np.testing.assert_array_equal(views[0]["camera_poses"][0, :3, 3], [4, 3, 2])
    assert views[0]["is_metric_scale"] is False
    images = conditioned_views([path, path], selected, "images")
    torch.testing.assert_close(images[0]["img"], views[0]["img"])
    assert "intrinsics" not in images[0] and "camera_poses" not in images[0]
