"""Ray-cast complete reconstruction surfaces with their captured colors."""

import cv2
import numpy as np
import open3d as o3d
import trimesh


class SurfaceRenderer:
    def __init__(self, mesh=None, texture=None, texture_uv=None):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.parts = {}
        if mesh is not None:
            self.add_mesh(mesh, texture, texture_uv)

    def add_mesh(self, mesh, texture=None, texture_uv=None):
        identifier = self.scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        )
        self.parts[identifier] = {
            "colors": np.asarray(mesh.vertex_colors),
            "triangles": np.asarray(mesh.triangles),
            "texture": texture,
            "texture_uv": texture_uv,
        }

    @classmethod
    def from_file(cls, path, glb=False, app_axes=True):
        if not glb:
            return cls(o3d.io.read_triangle_mesh(str(path)))
        source = trimesh.load_scene(path, process=False)
        fallback = None
        renderer = cls()
        for node in source.graph.nodes_geometry:
            transform, name = source.graph[node]
            app_mesh = source.geometry[name].copy()
            has_texture = (
                app_mesh.visual.kind == "texture"
                and getattr(app_mesh.visual.material, "baseColorTexture", None)
                is not None
            )
            if not has_texture:
                # Recover COLOR_0 per geometry instead of flattening mixed materials.
                if fallback is None:
                    fallback = trimesh.load_scene(
                        path, process=False, skip_materials=True
                    )
                app_mesh = fallback.geometry[name].copy()
            app_mesh.apply_transform(transform)
            if app_axes:
                app_mesh.apply_transform(np.diag([1.0, -1.0, -1.0, 1.0]))
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(app_mesh.vertices),
                o3d.utility.Vector3iVector(app_mesh.faces),
            )
            if has_texture:
                renderer.add_mesh(
                    mesh,
                    np.asarray(
                        app_mesh.visual.material.baseColorTexture.convert("RGB")
                    ),
                    np.asarray(app_mesh.visual.uv),
                )
            else:
                mesh.vertex_colors = o3d.utility.Vector3dVector(
                    np.asarray(app_mesh.visual.vertex_colors)[:, :3] / 255
                )
                renderer.add_mesh(mesh)
        if not renderer.parts:
            raise ValueError("No triangle geometry in the exported asset")
        return renderer

    def render(self, intrinsics, extrinsic, width, height):
        rays = self.scene.create_rays_pinhole(intrinsics, extrinsic, width, height)
        hit = self.scene.cast_rays(rays)
        visible = np.isfinite(hit["t_hit"].numpy())
        # Convert ray distance to camera Z instead of assuming normalized rays.
        ray_z = rays.numpy()[..., 3:] @ extrinsic[2, :3]
        rendered_depth = hit["t_hit"].numpy() * ray_z
        rendered = np.full((height, width, 3), 24, np.uint8)
        geometry_ids = hit["geometry_ids"].numpy()
        for identifier, part in self.parts.items():
            selected = visible & (geometry_ids == identifier)
            if not selected.any():
                continue
            primitive = hit["primitive_ids"].numpy()[selected]
            uv = hit["primitive_uvs"].numpy()[selected]
            weights = np.column_stack([1 - uv.sum(1), uv])
            texture, texture_uv, triangles, colors = (
                part[key] for key in ("texture", "texture_uv", "triangles", "colors")
            )
            if texture is not None:
                coordinates = np.sum(
                    texture_uv[triangles[primitive]] * weights[..., None],
                    axis=1,
                )
                sampled = np.empty((len(coordinates), 3), dtype=np.uint8)
                # OpenCV remap requires each destination dimension below SHRT_MAX.
                for start in range(0, len(coordinates), 32766):
                    batch = coordinates[start : start + 32766]
                    sampled[start : start + len(batch)] = cv2.remap(
                        texture,
                        (batch[:, 0] * texture.shape[1] - 0.5).astype(np.float32)[
                            :, None
                        ],
                        ((1 - batch[:, 1]) * texture.shape[0] - 0.5).astype(np.float32)[
                            :, None
                        ],
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )[:, 0]
                rendered[selected] = sampled
            else:
                rendered[selected] = (
                    (
                        255
                        * np.sum(
                            colors[triangles[primitive]] * weights[..., None],
                            axis=1,
                        )
                    )
                    .round()
                    .clip(0, 255)
                    .astype(np.uint8)
                )
        return rendered, rendered_depth, visible
