"""Constrain learned depth using independent feature-track cameras and sparse points."""
import json
from pathlib import Path

import cv2
import numpy as np

from .colmap_io import read_model
from .io import digest,write_json,write_npz


def refine(source,destination,colmap_model):
    source,destination=Path(source),Path(destination)
    images,points=read_model(colmap_model)
    if destination.exists():
        raise ValueError("Refinement output exists; choose a new directory to preserve the previous model")
    manifest=json.loads((source/"input.json").read_text())
    configuration=json.loads((source/"inference.json").read_text())
    windows=sorted((source/"windows").glob("*.npz"))
    from .inference import window_ranges
    expected=list(window_ranges(len(manifest["frames"]),configuration["window"],configuration["overlap"]))
    if [int(f.stem) for f in windows]!=[start for start,_ in expected]:
        raise ValueError("Finish source inference before camera refinement")
    destination.mkdir(parents=True)
    (destination/"frames").symlink_to((source/"frames").resolve(),target_is_directory=True)
    write_json(destination/"input.json",manifest)
    configuration.update({"poses_global":True,"pose_source":"COLMAP feature tracks and bundle adjustment",
                          "source_inference_sha256":digest(source/"inference.json"),
                          "colmap_images_sha256":digest(Path(colmap_model)/"images.txt")})
    write_json(destination/"inference.json",configuration)
    fits=[]
    for window in windows:
        data=dict(np.load(window))
        registered=np.zeros(len(data["frame_ids"]),bool)
        for index,frame in enumerate(data["frame_ids"]):
            name=Path(manifest["frames"][int(frame)]["file"]).name
            image=images.get(name)
            if image is None:continue
            depth=data["depth"][index]
            h,w=depth.shape
            camera=image["camera"]
            observations=image["observations"]
            valid=[o for o in observations if int(o[2]) in points and points[int(o[2])]["error"]<2 and points[int(o[2])]["track_length"]>=3]
            if len(valid)<30:continue
            valid=np.asarray(valid)
            xyz=np.array([points[int(o[2])]["xyz"] for o in valid])
            z=(xyz@image["extrinsics"][:,:3].T+image["extrinsics"][:,3])[:,2]
            u=(valid[:,0]*w/camera["width"]).astype(np.float32)
            v=(valid[:,1]*h/camera["height"]).astype(np.float32)
            predicted=cv2.remap(depth,u[:,None],v[:,None],cv2.INTER_LINEAR).ravel()
            keep=(z>0)&(predicted>0)
            if keep.sum()<30:continue
            ratio=z[keep]/predicted[keep]
            scale=float(np.median(ratio))
            error=np.abs(predicted[keep]*scale-z[keep])/z[keep]
            fit={"frame":int(frame),"window":int(window.stem),"sparse_points":int(keep.sum()),
                 "depth_scale":scale,"median_sparse_relative_depth_error":float(np.median(error)),
                 "p95_sparse_relative_depth_error":float(np.quantile(error,.95))}
            fit["accepted"] = fit["median_sparse_relative_depth_error"]<.15
            fits.append(fit)
            if not fit["accepted"]:continue
            intrinsic=camera["intrinsics"].copy()
            intrinsic[0]*=w/camera["width"];intrinsic[1]*=h/camera["height"]
            data["depth"][index]*=scale
            data["intrinsics"][index]=intrinsic
            data["extrinsics"][index]=image["extrinsics"]
            registered[index]=True
        data["registered"]=registered
        write_npz(destination/"windows"/window.name,**data)
    write_json(destination/"depth-calibration.json",{"fits":fits,"metric_scale_known":False,
        "registered_images":len(images),"source":str(source.resolve()),
        "method":"Per-frame median depth scale from triangulated feature tracks; globally bundle-adjusted poses"})
