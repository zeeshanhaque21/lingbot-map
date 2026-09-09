"""Local mesh and source-camera inspection."""

import json
import time
from pathlib import Path

import numpy as np
import viser


def view(output, port=8081):
    output = Path(output)
    if (
        not (output / "model" / "validation.json").exists()
        and (output / "pipeline-result.json").exists()
    ):
        output = Path(
            json.loads((output / "pipeline-result.json").read_text())["output"]
        )
    root = output / "model"
    report = json.loads((root / "validation.json").read_text())
    cameras = json.loads((root / "cameras.json").read_text())
    server = viser.ViserServer(host="127.0.0.1", port=port)
    server.scene.set_up_direction((0, -1, 0))
    asset_path = next(
        root / name
        for name in ("property-textured.glb", "property-unlit.glb", "property.glb")
        if (root / name).exists()
    )
    # Serve the exact GLB bytes; a parse/export roundtrip can discard vertex colors.
    server.scene.add_glb("/property", asset_path.read_bytes(), wxyz=(0, 1, 0, 0))
    positions = np.array([c["camera_to_world"] for c in cameras])[:, :3, 3]
    path = np.stack([positions[:-1], positions[1:]], axis=1)
    trajectory = server.scene.add_line_segments(
        "/capture_path", points=path, colors=(40, 190, 230), line_width=2
    )
    server.gui.configure_theme(control_layout="collapsible", dark_mode=True)
    server.gui.add_markdown(
        "## Property reconstruction\nObserved surfaces from the walkthrough.\n\n**Dimensions unverified.** Hidden surfaces remain open."
    )
    server.gui.add_markdown(
        f"{report['web_triangles']:,} display triangles · {len(cameras):,} views"
    )
    show_path = server.gui.add_checkbox("Capture path", initial_value=False)
    trajectory.visible = False

    @show_path.on_update
    def toggle(_):
        trajectory.visible = show_path.value

    selected = server.gui.add_slider(
        "Captured view",
        min=0,
        max=len(cameras) - 1,
        step=1,
        initial_value=min(14, len(cameras) - 1),
    )
    preview = server.gui.add_image(
        np.zeros((294, 518, 3), np.uint8), label="Captured view"
    )
    timestamp = server.gui.add_markdown("")
    move = server.gui.add_button("Compare from this camera")
    download = server.gui.add_button("Download GLB")

    @download.on_click
    def save(event):
        if event.client is not None:
            event.client.send_file_download("property.glb", asset_path.read_bytes())

    cached_window, archived_rgb, archived_ids = None, None, None

    def update():
        nonlocal cached_window, archived_rgb, archived_ids
        camera = cameras[selected.value]
        if cached_window != camera["window"]:
            windows = sorted((output / "windows").glob("*.npz"))
            with np.load(windows[camera["window"]]) as data:
                archived_rgb, archived_ids = data["rgb"], data["frame_ids"]
            cached_window = camera["window"]
        index = int(np.flatnonzero(archived_ids == camera["frame"])[0])
        preview.image = archived_rgb[index]
        timestamp.content = f"Capture time: {camera['timestamp_seconds']:.1f}s"

    @selected.on_update
    def select(_):
        update()

    @move.on_click
    def compare(event):
        if event.client is None:
            return
        camera = cameras[selected.value]
        pose = np.asarray(camera["camera_to_world"])
        event.client.camera.position = pose[:3, 3]
        event.client.camera.look_at = pose[:3, 3] + pose[:3, 2]
        event.client.camera.up_direction = -pose[:3, 1]
        event.client.camera.fov = 2 * np.arctan(
            preview.image.shape[0] / (2 * camera["intrinsics"][1][1])
        )

    @server.on_client_connect
    def connected(client):
        camera = cameras[selected.value]
        pose = np.asarray(camera["camera_to_world"])
        client.camera.position = pose[:3, 3]
        client.camera.look_at = pose[:3, 3] + pose[:3, 2]
        client.camera.up_direction = -pose[:3, 1]
        client.camera.fov = 2 * np.arctan(
            preview.image.shape[0] / (2 * camera["intrinsics"][1][1])
        )

    update()
    print(f"viewer: http://127.0.0.1:{port}", flush=True)
    while True:
        time.sleep(1)
