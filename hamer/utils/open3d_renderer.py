import open3d as o3d
import numpy as np
import torch
from typing import List, Optional

class Open3DRenderer:
    def __init__(self, faces: np.array):
        """
        Renderer using Open3D for headless rendering.
        Args:
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """
        self.faces = faces

    def render(self, vertices: np.array, camera_translation: np.array, mesh_base_color=(1.0, 1.0, 0.9)) -> np.array:
        """
        Render a mesh using Open3D.
        Args:
            vertices (np.array): Array of shape (V, 3) containing the mesh vertices.
            camera_translation (np.array): Array of shape (3,) with the camera translation.
            mesh_base_color (tuple): Base color of the mesh.
        Returns:
            np.array: Rendered image as a numpy array.
        """
        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        mesh.paint_uniform_color(mesh_base_color)

        # Apply camera translation
        mesh.translate(camera_translation)

        # Create a visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)  # Headless mode
        vis.add_geometry(mesh)

        # Render the scene
        vis.poll_events()
        vis.update_renderer()

        # Capture the image
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()

        # Convert to numpy array
        image_np = np.asarray(image)
        return image_np

    def render_multiple(self, vertices_list: List[np.array], camera_translations: List[np.array], mesh_base_color=(1.0, 1.0, 0.9)) -> np.array:
        """
        Render multiple meshes using Open3D.
        Args:
            vertices_list (List[np.array]): List of vertex arrays for each mesh.
            camera_translations (List[np.array]): List of camera translations for each mesh.
            mesh_base_color (tuple): Base color of the meshes.
        Returns:
            np.array: Rendered image as a numpy array.
        """
        # Create a visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)  # Headless mode

        for vertices, translation in zip(vertices_list, camera_translations):
            # Create Open3D mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(self.faces)
            mesh.paint_uniform_color(mesh_base_color)

            # Apply camera translation
            mesh.translate(translation)

            # Add mesh to visualizer
            vis.add_geometry(mesh)

        # Render the scene
        vis.poll_events()
        vis.update_renderer()

        # Capture the image
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()

        # Convert to numpy array
        image_np = np.asarray(image)
        return image_np 