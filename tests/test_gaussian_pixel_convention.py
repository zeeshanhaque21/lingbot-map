import numpy as np
import open3d as o3d

from lingbot_map.reconstruction.geometry import unproject


def physical_plane(offset):
    y, x = np.indices((17, 17))
    k = np.array([[60.0, 0, 8.5], [0, 65.0, 8.5], [0, 0, 1]])
    z = (
        2 / (1 - 0.4 * (x + offset - 8.5) / 60 + 0.2 * (y + offset - 8.5) / 65)
    ).astype(np.float32)
    return z, k


def test_gaussian_pixel_centers_recover_physical_plane():
    depth, k = physical_plane(0.5)
    xyz = unproject(depth, k, np.eye(4)[:3], pixel_center_offset=0.5)
    np.testing.assert_allclose(xyz @ [-0.4, 0.2, 1], 2, atol=2e-7)
    old = unproject(depth, k, np.eye(4)[:3])
    assert np.max(np.abs(old @ [-0.4, 0.2, 1] - 2)) > 0.003


def test_open3d_rgbd_and_gaussian_rays_agree():
    depth, k = physical_plane(0.5)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(17, 17, 60, 65, 8, 8)
    cloud = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth), intrinsic, depth_scale=1.0, depth_trunc=5.0
    )
    xyz = unproject(depth, k, np.eye(4)[:3], pixel_center_offset=0.5)
    np.testing.assert_allclose(np.asarray(cloud.points), xyz.reshape(-1, 3), atol=1e-7)


def test_existing_integer_pixel_convention_is_preserved():
    depth, k = physical_plane(0)
    xyz = unproject(depth, k, np.eye(4)[:3])
    np.testing.assert_allclose(xyz @ [-0.4, 0.2, 1], 2, atol=2e-7)
