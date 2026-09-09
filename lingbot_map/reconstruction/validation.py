"""Source-view render comparisons and explicit quality gates."""
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

from .io import write_json


def validate(output, maximum_views=24):
    output=Path(output)
    root=output/"model"
    cameras=json.loads((root/"cameras.json").read_text())
    manifest=json.loads((output/"input.json").read_text())
    mesh=o3d.io.read_triangle_mesh(str(root/"observed-surfaces.ply"))
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    colors=np.asarray(mesh.vertex_colors)
    triangles=np.asarray(mesh.triangles)
    selected=[c for c in cameras if c.get("held_out_from_fusion")]
    if not selected:
        selected=cameras
    selected=[selected[i] for i in np.linspace(0,len(selected)-1,min(maximum_views,len(selected)),dtype=int)]
    rows,metrics=[],[]
    cached_window=None
    for camera in selected:
        window=camera["window"]
        if cached_window!=window:
            files=sorted((output/"windows").glob("*.npz"))
            data=dict(np.load(files[window]));cached_window=window
        frame=camera["frame"]
        i=int(np.flatnonzero(data["frame_ids"]==frame)[0])
        reference=data["rgb"][i]
        h,w=reference.shape[:2]
        extrinsic=np.linalg.inv(np.asarray(camera["camera_to_world"]))
        rays=scene.create_rays_pinhole(np.asarray(camera["intrinsics"]),extrinsic,w,h)
        hit=scene.cast_rays(rays)
        visible=np.isfinite(hit["t_hit"].numpy())
        primitive=hit["primitive_ids"].numpy()[visible]
        uv=hit["primitive_uvs"].numpy()[visible]
        weights=np.column_stack([1-uv.sum(1),uv])
        rendered=np.full(reference.shape,24,np.uint8)
        if visible.any():
            rendered[visible]=(255*np.sum(colors[triangles[primitive]]*weights[...,None],axis=1)).clip(0,255).astype(np.uint8)
        difference=np.abs(rendered.astype(float)-reference)/255
        heat=np.zeros((h,w,3),np.uint8)
        heat[visible]=cv2.applyColorMap((difference.mean(-1)*255).astype(np.uint8),cv2.COLORMAP_INFERNO)[...,::-1][visible]
        row=Image.new("RGB",(w*3,h+32),(24,24,24))
        for j,(title,array) in enumerate([(f"Source {frame} · {camera['timestamp_seconds']:.1f}s",reference),
                                         ("Reconstructed visible surfaces",rendered),("Color difference; black = unobserved",heat)]):
            row.paste(Image.fromarray(array),(j*w,32));ImageDraw.Draw(row).text((j*w+8,8),title,fill="white")
        rows.append(row)
        metrics.append({"frame":frame,"held_out_from_fusion":camera.get("held_out_from_fusion",False),
                        "visible_mesh_fraction":float(visible.mean()),
                        "mean_absolute_rgb_error_visible":float(difference[visible].mean()) if visible.any() else None})
    for start in range(0,len(rows),6):
        sheet=Image.new("RGB",(rows[0].width,rows[0].height*len(rows[start:start+6])),(24,24,24))
        for j,row in enumerate(rows[start:start+6]):sheet.paste(row,(0,j*row.height))
        sheet.save(root/f"source-comparison-{start//6:02d}.jpg",quality=92)
    coverage=float(np.median([m["visible_mesh_fraction"] for m in metrics]))
    report={"views":metrics,"median_rendered_coverage":coverage,
            "metric_accuracy_verified":False,"visual_completeness_gate":coverage>=.7,
            "ready_for_verified_property_listing":False,
            "interpretation":"Frames withheld from TSDF fusion still participate in learned pose/depth inference. This tests view consistency, not ground-truth building dimensions.",
            "required_external_checks":["Measured scale and independent held-out lengths", "Room connectivity and loop drift", "Missing wall, doorway, glass and ceiling review"]}
    write_json(root/"render-validation.json",report)
    return report
