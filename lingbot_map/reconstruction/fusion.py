"""Fuse only multi-view-supported depth into a colored, open surface mesh."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
import trimesh

from .geometry import align_overlap, supported_depth
from .io import digest, write_json, write_npz


def fuse(output):
    output = Path(output)
    files = sorted((output/"windows").glob("*.npz"))
    if not files:
        raise ValueError("No inference windows; run infer first")
    manifest = json.loads((output/"input.json").read_text())
    inference = json.loads((output/"inference.json").read_text())
    expected = len(manifest["frames"])
    from .inference import window_ranges
    ranges = list(window_ranges(expected,inference["window"],inference["overlap"]))
    if [int(p.stem) for p in files] != [start for start,_ in ranges]:
        raise ValueError("Incomplete inference; resume infer before final fusion")
    artifact = output/"model"
    if (artifact/"validation.json").exists():
        raise ValueError("Model already exists; use its artifacts or a new output directory")
    artifact.mkdir(exist_ok=True)
    transforms, registrations = [np.eye(4)], []
    previous = dict(np.load(files[0]))
    voxel = float(np.median(previous["depth"])/180)
    for path in files[1:]:
        current = dict(np.load(path))
        transform, stats = align_overlap(previous,current)
        transforms.append(transforms[-1]@transform)
        registrations.append(stats)
        previous = current
    del previous
    write_json(artifact/"alignment.json", {"transforms": [t.tolist() for t in transforms],
        "overlaps": registrations, "method": "robust same-pixel Sim3 on shared frames",
        "loop_closure": False})
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,sdf_trunc=voxel*4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    diagnostics, cameras = [], []
    started = time.monotonic()
    for chunk,(path,transform) in enumerate(zip(files,transforms)):
        data = dict(np.load(path))
        scale = float(np.cbrt(np.linalg.det(transform[:3,:3])))
        rotation = transform[:3,:3]/scale
        start,end = ranges[chunk]
        # Prefer predictions after the scale bootstrap; use each frame exactly once.
        low = start if chunk==0 else (start+ranges[chunk-1][1])//2
        high = end if chunk==len(files)-1 else (end+ranges[chunk+1][0])//2
        for i,frame_id in enumerate(data["frame_ids"]):
            if not low <= frame_id < high:
                continue
            filtered,stats = supported_depth(i,data)
            filtered *= scale
            extrinsic = data["extrinsics"][i].astype(np.float64)
            local_c2w = np.eye(4)
            local_c2w[:3,:3] = extrinsic[:,:3].T
            local_c2w[:3,3] = -extrinsic[:,:3].T@extrinsic[:,3]
            global_c2w = np.eye(4)
            global_c2w[:3,:3] = rotation@local_c2w[:3,:3]
            global_c2w[:3,3] = scale*rotation@local_c2w[:3,3]+transform[:3,3]
            k = data["intrinsics"][i]
            h,w = filtered.shape
            intrinsic = o3d.camera.PinholeCameraIntrinsic(w,h,float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2]))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(data["rgb"][i])),
                o3d.geometry.Image(filtered),depth_scale=1.,depth_trunc=float(filtered.max()+1),
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd,intrinsic,np.linalg.inv(global_c2w))
            cameras.append({"frame": int(frame_id),"timestamp_seconds": manifest["frames"][int(frame_id)]["timestamp_seconds"],
                            "camera_to_world":global_c2w.tolist(),"intrinsics":k.tolist(),"window":chunk})
            diagnostics.append(stats)
        write_json(artifact/"progress.json",{"windows_fused":chunk+1,"frames_fused":len(cameras)})
        print(f"Fused {len(cameras)}/{expected} frames",file=sys.stderr,flush=True)
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if len(mesh.triangles)==0:
        raise ValueError("No supported surfaces survived fusion; inspect depth/pose diagnostics")
    o3d.io.write_triangle_mesh(str(artifact/"observed-surfaces.ply"),mesh)
    cloud = volume.extract_point_cloud()
    o3d.io.write_point_cloud(str(artifact/"observed-points.ply"),cloud)
    web_mesh = mesh.simplify_quadric_decimation(300000) if len(mesh.triangles)>300000 else mesh
    vertices = np.asarray(web_mesh.vertices).copy()
    # glTF is Y-up; rotate OpenCV world Y/Z to match. The source PLY stays OpenCV.
    vertices[:,1:] *= -1
    asset = trimesh.Trimesh(vertices=vertices,faces=np.asarray(web_mesh.triangles),
                            vertex_colors=(np.asarray(web_mesh.vertex_colors)*255).round().astype(np.uint8),process=False)
    asset.metadata.update({"units":"uncalibrated model units", "evidence":"multi-view supported depth",
                           "metric_accuracy":"unverified", "unseen_surfaces":"not completed"})
    asset.export(artifact/"property.glb")
    write_json(artifact/"cameras.json",cameras)
    write_json(artifact/"frame-validation.json",diagnostics)
    report = {"status":"reconstructed_unverified_metric_accuracy", "source_sha256":manifest["configuration"]["source_sha256"],
              "frames_expected":expected,"frames_fused":len(cameras),"temporal_coverage":len(cameras)/expected,
              "vertices":len(mesh.vertices),"triangles":len(mesh.triangles),"web_triangles":len(web_mesh.triangles),
              "voxel_size_model_units":voxel,"units":"uncalibrated model units",
              "accepted_pixel_fraction":sum(x["accepted_pixels"] for x in diagnostics)/sum(x["pixels"] for x in diagnostics),
              "observed_points":len(cloud.points),"fusion_seconds":time.monotonic()-started,
              "loop_closure":False,"absolute_dimensions_verified":False,"unseen_surfaces_completed":False,
              "limitations":["Monocular scale lacks measured anchors","Sequential window alignment can drift",
                             "Glass, mirrors, motion and unobserved surfaces remain uncertain",
                             "Depth consistency is an internal check, not independent ground truth"],
              "files":{p.name:digest(p) for p in artifact.glob("*") if p.suffix in (".glb",".ply")}}
    write_json(artifact/"validation.json",report)
