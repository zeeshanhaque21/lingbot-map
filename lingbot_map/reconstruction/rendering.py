"""Ray-cast complete reconstruction surfaces with their captured colors."""

import cv2
import numpy as np
import open3d as o3d
import trimesh


class SurfaceRenderer:
    def __init__(self, mesh, texture=None, texture_uv=None):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        self.colors = np.asarray(mesh.vertex_colors)
        self.triangles = np.asarray(mesh.triangles)
        self.texture = texture
        self.texture_uv = texture_uv

    @classmethod
    def from_file(cls, path, glb=False):
        if not glb:
            return cls(o3d.io.read_triangle_mesh(str(path)))
        app_mesh = trimesh.load(path, force="mesh", process=False)
        has_texture = (
            app_mesh.visual.kind == "texture"
            and getattr(app_mesh.visual.material, "baseColorTexture", None) is not None
        )
        if not has_texture:
            # Trimesh drops COLOR_0 when a material exists without a texture.
            app_mesh = trimesh.load(
                path, force="mesh", process=False, skip_materials=True
            )
        app_mesh.apply_transform(np.diag([1.0, -1.0, -1.0, 1.0]))
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(app_mesh.vertices),
            o3d.utility.Vector3iVector(app_mesh.faces),
        )
        if has_texture:
            return cls(
                mesh,
                np.asarray(app_mesh.visual.material.baseColorTexture.convert("RGB")),
                np.asarray(app_mesh.visual.uv),
            )
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(app_mesh.visual.vertex_colors)[:, :3] / 255
        )
        return cls(mesh)

    def render(self, intrinsics, extrinsic, width, height):
        rays = self.scene.create_rays_pinhole(intrinsics, extrinsic, width, height)
        hit = self.scene.cast_rays(rays)
        visible = np.isfinite(hit["t_hit"].numpy())
        # Convert ray distance to camera Z instead of assuming normalized rays.
        ray_z = rays.numpy()[..., 3:] @ extrinsic[2, :3]
        rendered_depth = hit["t_hit"].numpy() * ray_z
        primitive = hit["primitive_ids"].numpy()[visible]
        uv = hit["primitive_uvs"].numpy()[visible]
        weights = np.column_stack([1 - uv.sum(1), uv])
        rendered = np.full((height, width, 3), 24, np.uint8)
        if visible.any():
            if self.texture is not None:
                coordinates = np.sum(
                    self.texture_uv[self.triangles[primitive]] * weights[..., None],
                    axis=1,
                )
                sampled = np.empty((len(coordinates), 3), dtype=np.uint8)
                # OpenCV remap requires each destination dimension below SHRT_MAX.
                for start in range(0, len(coordinates), 32766):
                    batch = coordinates[start : start + 32766]
                    sampled[start : start + len(batch)] = cv2.remap(
                        self.texture,
                        (batch[:, 0] * self.texture.shape[1] - 0.5).astype(np.float32)[
                            :, None
                        ],
                        ((1 - batch[:, 1]) * self.texture.shape[0] - 0.5).astype(
                            np.float32
                        )[:, None],
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )[:, 0]
                rendered[visible] = sampled
            else:
                rendered[visible] = (
                    (
                        255
                        * np.sum(
                            self.colors[self.triangles[primitive]] * weights[..., None],
                            axis=1,
                        )
                    )
                    .clip(0, 255)
                    .astype(np.uint8)
                )
        return rendered, rendered_depth, visible
