import numpy as np
import pickle
import open3d as o3d
# demo_out/ego.png_out.npz

from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
import torch
import cv2
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from test_render import render_mano
import trimesh

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)


def plot_alignment(transformed_source, target, name, title=None):
    """
    Plot the 3D meshes of the transformed source and target in the same plot with different colors.
    
    Parameters:
        transformed_source (np.ndarray): Transformed source vertices (shape: [N,3])
        target (np.ndarray): Target vertices (shape: [N,3])
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    # Assumes that transformed_source and target are concatenated hands 
    # and that faces and faces_left are loaded globally.
    half = transformed_source.shape[0] // 2
    both_hands_faces = np.concatenate([faces, faces_left + half], axis=0)

    # Create a larger figure for increased resolution
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot transformed_source mesh (red) with thinner edges
    faces_vertices_source = transformed_source[both_hands_faces]
    poly_source = Poly3DCollection(faces_vertices_source, alpha=0.5, edgecolor='k', linewidths=0.5)
    poly_source.set_facecolor((0.9, 0.2, 0.2))  # red color
    ax.add_collection3d(poly_source)

    # Plot target mesh (blue) with thinner edges
    faces_vertices_target = target[both_hands_faces]
    poly_target = Poly3DCollection(faces_vertices_target, alpha=0.5, edgecolor='k', linewidths=0.5)
    poly_target.set_facecolor((0.2, 0.2, 0.9))  # blue color
    ax.add_collection3d(poly_target)

    # Adjust plot limits based on both meshes and add a margin to ensure they're fully in frame
    all_points = np.concatenate([transformed_source, target], axis=0)
    x_limits = [np.min(all_points[:, 0]), np.max(all_points[:, 0])]
    y_limits = [np.min(all_points[:, 1]), np.max(all_points[:, 1])]
    z_limits = [np.min(all_points[:, 2]), np.max(all_points[:, 2])]
    
    # Add a 10% margin around the limits
    margin_x = (x_limits[1] - x_limits[0]) * 0.1
    margin_y = (y_limits[1] - y_limits[0]) * 0.1
    margin_z = (z_limits[1] - z_limits[0]) * 0.1
    
    ax.set_xlim([x_limits[0] - margin_x, x_limits[1] + margin_x])
    ax.set_ylim([y_limits[0] - margin_y, y_limits[1] + margin_y])
    ax.set_zlim([z_limits[0] - margin_z, z_limits[1] + margin_z])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        plt.title(title)
    plt.savefig(name, dpi=300)
    plt.close()

def umeyama_alignment(source, target):
    """
    Umeyama algorithm to find rotation, translation and scaling between two point sets.
    
    Parameters:
        source (np.ndarray): (N, D) array of source points.
        target (np.ndarray): (N, D) array of target points.
    
    Returns:
        R (np.ndarray): (D, D) rotation matrix.
        t (np.ndarray): (D,) translation vector.
        s (float): Scaling factor.
        loss (float): Mean squared error between the aligned source and target points.
    """
    # Get number of points and dimensions
    n, d = source.shape
    
    # Center the points
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    
    # Center the points
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    
    # Covariance matrix
    cov = target_centered.T @ source_centered / n
    
    # SVD of covariance matrix
    u, s_vals, vh = np.linalg.svd(cov)
    
    # Handle reflection case
    reflection_matrix = np.eye(d)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        reflection_matrix[d-1, d-1] = -1
    
    # Calculate rotation
    R = u @ reflection_matrix @ vh
    
    # Calculate scaling
    var_source = np.sum(np.var(source_centered, axis=0))
    s = 1.0 if var_source == 0 else np.sum(s_vals) / var_source
    
    # Calculate translation
    t = target_centroid - s * R @ source_centroid

    # Compute alignment error (loss) as mean squared error between transformed source and target
    transformed_source = s * (R @ source.T).T + t
    loss = np.mean(np.sum((target - transformed_source) ** 2, axis=1))
    
    return R, t, s, loss, transformed_source

def np_dict2torch_dict(np_dict):
    torch_dict = {}
    for key, value in np_dict.items():
        torch_dict[key] = torch.from_numpy(value)
    return torch_dict

# load the model
model_cfg = pickle.load(open('model_cfg.pkl', 'rb'))
print("model_cfg loaded")

# load the faces
faces = pickle.load(open('model_mano_faces.pkl', 'rb'))
faces_left = faces[:,[0,2,1]]
print("faces loaded")


def vertices_to_trimesh(vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9), 
                        rot_axis=[1,0,0], rot_angle=0, is_right=1):
    # material = pyrender.MetallicRoughnessMaterial(
    #     metallicFactor=0.0,
    #     alphaMode='OPAQUE',
    #     baseColorFactor=(*mesh_base_color, 1.0))
    vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
    if is_right:
        mesh = trimesh.Trimesh(vertices.copy() + camera_translation, faces.copy(), vertex_colors=vertex_colors)
    else:
        mesh = trimesh.Trimesh(vertices.copy() + camera_translation, faces_left.copy(), vertex_colors=vertex_colors)
    # mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
    
    rot = trimesh.transformations.rotation_matrix(
            np.radians(rot_angle), rot_axis)
    mesh.apply_transform(rot)

    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    return mesh

def both_hands_verts_to_trimesh(verts, mesh_base_color=(1.0, 1.0, 0.9)):
    print("verts: ", verts.shape)
    print("max face: ", faces.max())
    vertex_colors = np.array([(*mesh_base_color, 1.0)] * verts.shape[0])
    both_hands_faces = np.concatenate([faces, faces_left + verts.shape[0] // 2], axis=0)
    print("max face: ", both_hands_faces.max())
    mesh = trimesh.Trimesh(verts, both_hands_faces, vertex_colors=vertex_colors)
    return mesh

def concat_both_hands_verts(verts, is_right):
    # first hand has to be right hand
    if is_right[0]:
        first = verts[0]
        second = verts[1]
    else:
        first = verts[1]
        second = verts[0]
    return np.concatenate([first, second], axis=0)


def create_vertices_wo_translation(out, batch):
    verts = out['pred_vertices']
    is_right = batch['right']
    multiplier = (2 * is_right - 1).reshape(-1, 1)  # Shape (2, 1)
    verts[:, :, 0] = multiplier * verts[:, :, 0]  
    return verts

def translate_verts(verts, focal_length, out, batch):
    multiplier = (2*batch['right']-1)
    pred_cam = out['pred_cam']
    pred_cam[:,1] = multiplier*pred_cam[:,1]
    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    img_size = batch["img_size"].float()
    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, focal_length).detach().cpu().numpy()
    return verts + pred_cam_t_full[:,None,:]

def find_optimal_focal_length(out1, out2, batch1, batch2):
    import csv
    with open('focal_length_loss.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['focal_length', 'loss'])
        verts1 = create_vertices_wo_translation(out1, batch1)
        verts2 = create_vertices_wo_translation(out2, batch2)
        
        for focal_length in range(200, 400, 50):
            print(f"focal_length: {focal_length}")
            verts_cp1 = verts1.copy()
            verts_cp2 = verts2.copy()
            verts_cp1 = translate_verts(verts_cp1, focal_length, out1, batch1)
            verts_cp1 = concat_both_hands_verts(verts_cp1, batch1['right'])
            verts_cp2 = translate_verts(verts_cp2, focal_length, out2, batch2)
            verts_cp2 = concat_both_hands_verts(verts_cp2, batch2['right'])
            mesh1 = both_hands_verts_to_trimesh(verts_cp1)
            mesh2 = both_hands_verts_to_trimesh(verts_cp2)
            mesh1.export(f"mesh1_{focal_length}.obj")
            mesh2.export(f"mesh2_{focal_length}.obj")
            # try aligining the verts
            R, t, s, loss, transformed_source = umeyama_alignment(verts_cp1, verts_cp2)
            plot_alignment(transformed_source, verts_cp2, f"focal_length_alignment_{focal_length}.png", title=f"Focal Length: {focal_length}")

            print(f"loss: {loss}")
            writer.writerow([focal_length, loss])

def draw_optimal_focal_length_graph():
    import matplotlib.pyplot as plt
    import csv

    # Read the CSV file
    with open('focal_length_loss.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        focal_lengths = []
        losses = []
        for row in reader:
            focal_lengths.append(float(row[0]))
            losses.append(float(row[1]))
            
    plt.plot(focal_lengths, losses)
    plt.title("Loss vs Focal Length")
    plt.xlabel("Focal Length")
    plt.ylabel("Loss")
    plt.savefig('focal_length_loss.png')
    
    # print the minimum loss and the focal length at which it occurs
    min_loss = min(losses)
    min_loss_index = losses.index(min_loss)
    print(f"Minimum loss: {min_loss} at focal length: {focal_lengths[min_loss_index]}")

def create_mesh(out, batch, name):
    multiplier = (2*batch['right']-1)
    pred_cam = out['pred_cam']
    pred_cam[:,1] = multiplier*pred_cam[:,1]
    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    img_size = batch["img_size"].float()
    multiplier = (2*batch['right']-1)
    

    
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

    verts = out['pred_vertices']
    is_right = batch['right']

    multiplier = (2 * is_right - 1).reshape(-1, 1)  # Shape (2, 1)
    verts[:, :, 0] = multiplier * verts[:, :, 0]  

    first = vertices_to_trimesh(verts[0], pred_cam_t_full[0], LIGHT_BLUE, [1,0,0], 0, is_right=is_right[0])
    second = vertices_to_trimesh(verts[1], pred_cam_t_full[1], LIGHT_BLUE, [1,0,0], 0, is_right=is_right[1])

    mesh_list = [first, second]
    # save the meshes as single obj file
    mesh = trimesh.util.concatenate(mesh_list)
    mesh.export(f"{name}.obj")

# # load the output
out1 = np.load('demo_out/ego.png_out.npz')
batch1 = np.load('demo_out/ego.png_batch.npz')

out2 = np.load('demo_out/exo.png_out.npz')
batch2 = np.load('demo_out/exo.png_batch.npz')

batch1 = np_dict2torch_dict(batch1)
batch2 = np_dict2torch_dict(batch2)
# out = np_dict2torch_dict(out)


verts = find_optimal_focal_length(out1, out2, batch1, batch2)

draw_optimal_focal_length_graph()

# img_size = batch1["img_size"].float()

# print("img_size: ", img_size)

# scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()

# print(model_cfg.MODEL.IMAGE_SIZE)
# print(model_cfg.EXTRA.FOCAL_LENGTH)

########################


# create_mesh(out, batch, "mesh_ego")


# out = np.load('demo_out/exo.png_out.npz')
# batch = np.load('demo_out/exo.png_batch.npz')

# batch = np_dict2torch_dict(batch)
# # out = np_dict2torch_dict(out)

# create_mesh(out, batch, "mesh_exo")


######################

# batch_size = batch['img'].shape[0]



# verts_3d = out['pred_vertices']

# verts_left = verts_3d[0]
# verts_right = verts_3d[1]

# pred_cam_t_full_left = pred_cam_t_full[0]
# pred_cam_t_full_right = pred_cam_t_full[1]

# # translate verts 
# verts_left = verts_left + pred_cam_t_full_left
# verts_right = verts_right + pred_cam_t_full_right


# concat_verts = np.concatenate([verts_left, verts_right], axis=0)

# # save the keypoints as ply
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(concat_verts)
# o3d.io.write_point_cloud("verts.ply", point_cloud)