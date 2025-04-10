import pyrender
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"


def project_vertices(vertices, camera, camera_pose, width, height):
    projection_matrix = camera.get_projection_matrix(width=width, height=height)
    view_matrix = np.linalg.inv(camera_pose)
    mvp_matrix = projection_matrix @ view_matrix

    vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    clip_space_vertices = (mvp_matrix @ vertices_homogeneous.T).T

    w = clip_space_vertices[:, 3]
    valid_w_indices = w > 1e-5
    clip_space_vertices_valid = clip_space_vertices[valid_w_indices]
    w_valid = w[valid_w_indices]

    if clip_space_vertices_valid.shape[0] == 0:
        return np.empty((0, 2))

    ndc_vertices = clip_space_vertices_valid[:, :3] / w_valid[:, None]

    screen_coords = np.zeros((ndc_vertices.shape[0], 2))
    screen_coords[:, 0] = (ndc_vertices[:, 0] + 1) * width / 2
    screen_coords[:, 1] = (1 - ndc_vertices[:, 1]) * height / 2

    return screen_coords


# Create the mesh (cube)
mesh = trimesh.creation.box(extents=(1, 1, 1))
scene = pyrender.Scene()

# Move the mesh in front of the camera
mesh_pose = np.eye(4)
mesh_pose[2, 3] = -3  # Move along the Z-axis
mesh_node = scene.add(pyrender.Mesh.from_trimesh(mesh), pose=mesh_pose)

# Camera parameters
fx = 500
fy = 500
cx = 320
cy = 240
width = 640
height = 480

# Create an intrinsic camera
camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
camera_pose = np.eye(4)
camera_pose[2, 3] = 0  # Camera at origin, looking down negative z
camera_node = scene.add(camera, pose=camera_pose)

# Renderer setup
renderer = pyrender.OffscreenRenderer(width, height)

# Render the scene with wireframe
color, depth = renderer.render(
    scene,
    flags=pyrender.RenderFlags.RGBA
    | pyrender.RenderFlags.FLAT
    | pyrender.RenderFlags.SKIP_CULL_FACES
    | pyrender.RenderFlags.ALL_WIREFRAME,
)

# Save the rendered wireframe image
plt.figure()
plt.imshow(color)
plt.title("Rendered Scene (Wireframe)")
plt.savefig("test_wireframe.png")
plt.close()

# Project the mesh vertices using the function
vertices = mesh.vertices
# transfor vertices with mesh_pose
homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
vertices = (mesh_pose @ homogeneous_vertices.T).T
vertices = vertices[:, :3]


print("vertices", vertices)

screen_coords = project_vertices(vertices, camera, camera_pose, width, height)

print("Projected Screen Coordinates:")
print(screen_coords)
# Plot the projected vertices
plt.figure()  # Ensure a new figure is created
plt.scatter(screen_coords[:, 0], screen_coords[:, 1], c="red", s=10)  # Smaller points
plt.title("Projected Vertices onto 2D Screen")
plt.xlim(0, width)
plt.ylim(height, 0)  # Flip the y-axis to match image coordinates
plt.gca().set_aspect("equal", adjustable="box")
plt.grid(True)
plt.savefig("test_projected.png")
plt.close()

# plot color and scatter plot in same figure
plt.figure()
plt.imshow(color)
plt.scatter(screen_coords[:, 0], screen_coords[:, 1], c="red", s=10)
plt.xlim(0, width)
plt.ylim(height, 0)
plt.savefig("test_projected_color.png")
plt.close()
