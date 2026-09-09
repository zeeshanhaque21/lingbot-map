import numpy as np
from scipy.spatial.transform import Rotation

from lingbot_map.reconstruction.geometry import (
    robust_similarity, supported_depth, transform_points, unproject,
)
from lingbot_map.reconstruction.inference import window_ranges


def test_opencv_unprojection_uses_camera_z_and_world_to_camera():
    rotation=Rotation.from_euler("xyz",[.2,.1,-.3]).as_matrix()
    center=np.array([2.,3.,4.])
    extrinsic=np.column_stack([rotation,-rotation@center])
    k=np.array([[10.,0,1],[0,20.,1],[0,0,1.]])
    points=unproject(np.full((3,3),2.),k,extrinsic)
    np.testing.assert_allclose(points[1,1],center+rotation.T@np.array([0,0,2.]))
    camera=points@rotation.T+extrinsic[:,3]
    pixel=camera@k.T
    np.testing.assert_allclose(pixel[2,2,:2]/pixel[2,2,2],[2,2])


def test_registration_recovers_scale_and_rotation_with_outliers():
    rng=np.random.default_rng(42)
    source=rng.normal(size=(1000,3))
    truth=np.eye(4)
    truth[:3,:3]=1.7*Rotation.from_euler("xyz",[.2,-.4,.7]).as_matrix()
    truth[:3,3]=[3,-4,2]
    target=transform_points(source,truth)+rng.normal(0,.001,source.shape)
    target[:80]+=rng.normal(0,2,(80,3))
    fit,_=robust_similarity(source,target)
    np.testing.assert_allclose(fit,truth,atol=.003)


def test_multiview_filter_rejects_depth_inconsistent_patch():
    k=np.array([[30.,0,16],[0,30.,16],[0,0,1.]],np.float32)
    depth=np.full((5,32,32),2,np.float32)
    depth[2,8:16,8:16]=3
    rng=np.random.default_rng(1)
    data={"depth":depth,"intrinsics":np.repeat(k[None],5,0),
          "extrinsics":np.repeat(np.eye(4,dtype=np.float32)[None,:3],5,0),
          "confidence":rng.uniform(5,10,(5,32,32)).astype(np.float32),
          "frame_ids":np.arange(5)}
    filtered,stats=supported_depth(2,data)
    assert not filtered[8:16,8:16].any()
    assert stats["accepted_pixels"]>400


def test_window_ownership_has_full_coverage_without_duplicates():
    for count in (12,96,97,192,1000):
        ranges=list(window_ranges(count,96,24))
        owned=[]
        for i,(start,end) in enumerate(ranges):
            lo=start if i==0 else (start+ranges[i-1][1])//2
            hi=end if i==len(ranges)-1 else (end+ranges[i+1][0])//2
            owned.extend(range(lo,hi))
        assert owned==list(range(count))
