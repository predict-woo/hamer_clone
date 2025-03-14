import pickle
import numpy as np
import open3d as o3d

# load the model
model_cfg = pickle.load(open('model_cfg.pkl', 'rb'))
print("model_cfg loaded")

# load mano hand faces
mano_faces = pickle.load(open('model_mano_faces.pkl', 'rb'))
print("mano_faces loaded")


faces_new = np.array([[92, 38, 234],
                      [234, 38, 239],
                      [38, 122, 239],
                      [239, 122, 279],
                      [122, 118, 279],
                      [279, 118, 215],
                      [118, 117, 215],
                      [215, 117, 214],
                      [117, 119, 214],
                      [214, 119, 121],
                      [119, 120, 121],
                      [121, 120, 78],
                      [120, 108, 78],
                      [78, 108, 79]])

faces = np.concatenate([mano_faces, faces_new], axis=0)

focal_length = model_cfg.EXTRA.FOCAL_LENGTH
img_res = model_cfg.MODEL.IMAGE_SIZE

camera_center = [img_res // 2, img_res // 2]
faces_left = faces[:,[0,2,1]]

def render_mano(vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9)):
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces_left)
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
