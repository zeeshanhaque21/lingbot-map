"""Local mesh and source-camera inspection."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import trimesh
import viser


def view(output, port=8081):
    output = Path(output)
    root = output/"model"
    report = json.loads((root/"validation.json").read_text())
    cameras = json.loads((root/"cameras.json").read_text())
    manifest = json.loads((output/"input.json").read_text())
    server = viser.ViserServer(host="127.0.0.1",port=port)
    server.scene.set_up_direction((0,-1,0))
    mesh = trimesh.load(root/"observed-surfaces.ply",process=False)
    server.scene.add_mesh_trimesh("/property",mesh)
    positions = np.array([c["camera_to_world"] for c in cameras])[:,:3,3]
    path = np.stack([positions[:-1],positions[1:]],axis=1)
    trajectory = server.scene.add_line_segments("/capture_path",points=path,
                                                 colors=(40,190,230),line_width=2)
    server.gui.configure_theme(control_layout="collapsible",dark_mode=True)
    server.gui.add_markdown("## Property reconstruction\nObserved surfaces from the walkthrough.\n\n**Dimensions unverified.** Hidden surfaces remain open.")
    server.gui.add_markdown(f"{report['triangles']:,} triangles · {len(cameras):,} views")
    show_path = server.gui.add_checkbox("Capture path",initial_value=True)
    @show_path.on_update
    def toggle(_):
        trajectory.visible=show_path.value
    selected = server.gui.add_slider("Source frame",min=0,max=len(cameras)-1,step=1,initial_value=0)
    preview = server.gui.add_image(np.zeros((294,518,3),np.uint8),label="Captured view")
    timestamp = server.gui.add_markdown("")
    move = server.gui.add_button("Compare from this camera")
    download = server.gui.add_button("Download GLB")
    @download.on_click
    def save(event):
        if event.client is not None:
            event.client.send_file_download("property.glb",(root/"property.glb").read_bytes())
    def update():
        camera = cameras[selected.value]
        frame = manifest["frames"][camera["frame"]]
        preview.image = np.asarray(Image.open(output/frame["file"]).resize((518,294)))
        timestamp.content=f"Capture time: {camera['timestamp_seconds']:.1f}s"
    @selected.on_update
    def select(_):
        update()
    @move.on_click
    def compare(event):
        if event.client is None:
            return
        camera = cameras[selected.value]
        pose = np.asarray(camera["camera_to_world"])
        event.client.camera.position=pose[:3,3]
        event.client.camera.look_at=pose[:3,3]+pose[:3,2]
        event.client.camera.up_direction=-pose[:3,1]
        event.client.camera.fov=2*np.arctan(294/(2*camera["intrinsics"][1][1]))
    @server.on_client_connect
    def connected(client):
        extent=float(np.linalg.norm(mesh.extents))
        center=mesh.bounds.mean(0)
        client.camera.position=center+np.array([.5,-.7,-.7])*extent
        client.camera.look_at=center
        client.camera.up_direction=(0,-1,0)
    update()
    print(f"viewer: http://127.0.0.1:{port}",flush=True)
    while True:
        time.sleep(1)
