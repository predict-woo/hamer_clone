import os
import pickle
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
import copy
import open3d as o3d


faces = np.load('config/model_mano_faces.npy')
faces_left = faces[:,[0,2,1]]

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)


def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1,3)).transpose(1,0)).transpose(1,0)
    return world_coord


def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord


def hand2hand(points, R, t):
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    RT = np.hstack([R, t])
    camera_points_h = (RT @ points_h.T).T
    camera_points = camera_points_h[:, :3]
    return camera_points


def rotation2rotation(A, B):
    try:
        A_inv = A.T
        # X = np.dot(A_inv, B) # X = A-1 * B ~~ AX = B
        X = np.dot(B, A_inv) # X = B * A-1 ~~ XA = B
        return X
    except Exception as e:
        print(f"오류 발생: {e}")
        return None


def points2image(points, colors, RT, K, image_width, image_height):
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    camera_points_h = (RT @ points_h.T).T
    camera_points = camera_points_h[:, :3]
    
    projected_points = (K @ camera_points.T).T
    projected_points[:, 0] /= projected_points[:, 2]
    projected_points[:, 1] /= projected_points[:, 2]

    image_width, image_height = int(image_width), int(image_height)
    rgb_map = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    depths = camera_points[:, 2]
    indices = np.argsort(depths)[::-1]
    
    for i in indices:
        point = projected_points[i]
        x = int(round(point[0]))
        y = int(round(point[1]))
        z = depths[i]

        if 0 <= x < image_width and 0 <= y < image_height and z > 0:
            color = colors[i].astype(np.uint8).tolist()
            cv2.circle(rgb_map, (x, y), 0, color, -1)
    
    return rgb_map


def depth2points(depth, fx, fy, cx, cy):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.dstack((x, y, z))


def exo2ego(exo_rgb_path, exo_depth_path, 
            exo_cam_int_path, exo_cam_ext_path, exo_hand_path, 
            ego_cam_int_path, ego_cam_ext_path, ego_hand_path):
    
    # get pointcloud from exos
    exo_cam_int = np.loadtxt(exo_cam_int_path)
    exo_fx, exo_fy, exo_cx, exo_cy = exo_cam_int[:4]
    exo_cam_ext = np.loadtxt(exo_cam_ext_path).reshape(4, 4)
    exo_depth = cv2.imread(exo_depth_path, cv2.IMREAD_ANYDEPTH)
    points = depth2points(exo_depth, exo_fx, exo_fy, exo_cx, exo_cy).reshape(-1, 3)
    colors = cv2.imread(exo_rgb_path).reshape(-1, 3)
    
    # project ego with X
    ego_cam_ext = np.loadtxt(ego_cam_ext_path).reshape(4, 4)
    exo_R = exo_cam_ext[:3, :3]
    ego_R = ego_cam_ext[:3, :3]
    exo_t = exo_cam_ext[:3, 3:]
    ego_t = ego_cam_ext[:3, 3:]
    points /= 1000
    ego_cam_ext_inv = np.linalg.inv(ego_cam_ext)
    points_one = np.ones_like(points[:, 0])
    points_one = np.expand_dims(points_one, axis=1)
    points = np.hstack([points, points_one])
    translated_points = ego_cam_ext_inv @ exo_cam_ext @ points.T
    translated_points = translated_points.T[:, :3]
    X = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    ego_cam_int = np.loadtxt(ego_cam_int_path)
    ego_fx, ego_fy, ego_cx, ego_cy, ego_w, ego_h = ego_cam_int
    ego_K = np.array([[ego_fx, 0, ego_cx], 
                      [0, ego_fy, ego_cy],
                      [0, 0, 1]])
    ego_rgb_pred = points2image(translated_points, colors, X, ego_K, ego_w, ego_h)

    return ego_rgb_pred


def load_rgb_depth_cam_ext_int(root_path, subject, cam_id, index=None):
    rgb_paths = sorted(glob(os.path.join(root_path, subject, '*/*', cam_id, 'rgb/*.png')))
    depth_paths = [path.replace('rgb', 'depth') for path in rgb_paths]
    hand_paths = [path.replace('rgb', 'hand_pose').replace('png', 'txt') for path in rgb_paths]
    cam_ext_paths = [path.replace('rgb', 'cam_pose').replace('png', 'txt') for path in rgb_paths]
    cam_int_paths = [path.split('rgb')[0] + 'cam_intrinsics.txt' for path in rgb_paths]
    if index is not None:
        return rgb_paths[index], depth_paths[index], hand_paths[index], cam_ext_paths[index], cam_int_paths[index]
    return rgb_paths, depth_paths, hand_paths, cam_ext_paths, cam_int_paths

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

def umeyama_alignment(source_pcd, target_pcd):
    """
    Umeyama algorithm to find optimal transformation between two point clouds.
    
    Parameters:
        source_pcd (o3d.geometry.PointCloud): Source point cloud.
        target_pcd (o3d.geometry.PointCloud): Target point cloud.
    
    Returns:
        tuple: (transformation, loss, transformed_source_pcd)
            - transformation (np.ndarray): 4×4 transformation matrix.
            - loss (float): Mean squared error between the aligned source and target points.
            - transformed_source_pcd (o3d.geometry.PointCloud): The transformed source point cloud.
    """
    # Make copies to avoid modifying the originals
    source_pcd = copy.deepcopy(source_pcd)
    target_pcd = copy.deepcopy(target_pcd)
    
    # Get points as numpy arrays
    source = np.asarray(source_pcd.points)
    target = np.asarray(target_pcd.points)
    
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

    # Create a transformation matrix from R, t, and s
    transformation = np.eye(4)
    transformation[:3, :3] = s * R
    transformation[:3, 3] = t
    
    # Apply transformation to create the transformed source point cloud
    transformed_source_pcd = copy.deepcopy(source_pcd)
    transformed_source_pcd.transform(transformation)
    
    # Compute alignment error (loss) as mean squared error
    transformed_points = np.asarray(transformed_source_pcd.points)
    loss = np.mean(np.sum((target - transformed_points) ** 2, axis=1))
    
    return transformation, loss, transformed_source_pcd


def register_point_clouds(source_pcd, target_pcd, voxel_size=0.005, default_color=[0, 0.651, 0.929]):
    """
    Registers a source point cloud to a target point cloud using Mean Alignment + RANSAC + ICP.

    Args:
        source_pcd (o3d.geometry.PointCloud): Source point cloud to be aligned.
        target_pcd (o3d.geometry.PointCloud): Target point cloud to align to.
        voxel_size (float): Voxel size for downsampling and normal/feature estimation.
        default_color (list): RGB color (0-1 range) to paint the source cloud if it lacks colors.

    Returns:
        tuple: (transformed_source_pcd, final_transformation)
            - transformed_source_pcd (o3d.geometry.PointCloud): The source point cloud after alignment.
            - final_transformation (np.ndarray): The 4x4 transformation matrix combining all steps.
    """
    print(f"\nStarting registration with voxel size: {voxel_size}")

    # Make copies to avoid modifying the originals
    source_pcd = copy.deepcopy(source_pcd)
    target_pcd = copy.deepcopy(target_pcd)
    
    # --- Initial Mean Alignment ---
    print("Performing initial mean alignment...")
    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)
    
    source_mean = np.mean(source_points, axis=0)
    target_mean = np.mean(target_points, axis=0)
    translation_vector = target_mean - source_mean
    
    # Create a 4x4 transformation matrix for the mean alignment
    mean_alignment_transform = np.eye(4)
    mean_alignment_transform[:3, 3] = translation_vector
    
    # Apply translation directly to the source point cloud
    source_pcd.translate(translation_vector)
    print(f"Applied translation vector: {translation_vector}")
    # --- End Mean Alignment ---

    # Add uniform color if the point clouds don't have colors
    if not source_pcd.has_colors():
        source_pcd.paint_uniform_color(default_color)
    if not target_pcd.has_colors():
        target_pcd.paint_uniform_color(default_color)

    # --- Global Registration (RANSAC) ---
    print("Downsampling point clouds...")
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print("Estimating normals for RANSAC...")
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    print("Normals estimated for RANSAC.")

    radius_feature = voxel_size * 5
    print("Computing FPFH features...")
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    print("FPFH features computed.")

    distance_threshold_global = voxel_size * 1.5
    print("Running RANSAC...")
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold_global,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold_global)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    print("RANSAC finished.")
    print("Global registration (RANSAC) fitness:", result_ransac.fitness)
    print("Global registration (RANSAC) inlier_rmse:", result_ransac.inlier_rmse)

    if result_ransac.fitness < 0.1: # Add a check if RANSAC failed significantly
         print("Warning: RANSAC fitness is low. ICP might fail or be inaccurate.")
         # Optionally return None or raise an error if RANSAC fails badly
         # return None, np.identity(4)

    # --- Fine-tuning with ICP ---
    # Estimate normals for the original point clouds before ICP
    print("Estimating normals for ICP...")
    # Ensure normals are estimated for both, PointToPlane benefits from source normals too
    source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    print("Normals estimated for ICP.")

    distance_threshold_icp = voxel_size * 0.4
    print("Running ICP...")
    result_icp = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, distance_threshold_icp, result_ransac.transformation, # Use RANSAC result as initial guess
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print("ICP finished.")
    print("ICP Fitness:", result_icp.fitness)
    print("ICP Inlier RMSE:", result_icp.inlier_rmse)

    # ICP transformation (which already includes RANSAC as initial guess)
    icp_transformation = result_icp.transformation
    
    # Apply the ICP transformation to the source point cloud
    source_pcd.transform(icp_transformation)
    
    # Combine transformations: first mean alignment, then ICP
    # The order is important: final_T = T_icp @ T_mean
    final_transformation = np.matmul(icp_transformation, mean_alignment_transform)

    print("Registration complete.")
    return source_pcd, final_transformation


def depth2pcd(depth, rgb, cam_int, cam_ext, mask=None):
    fx, fy, cx, cy = cam_int[:4]
    # cam_ext = np.loadtxt(cam_ext_path).reshape(4, 4)
    points = depth2points(depth, fx, fy, cx, cy).reshape(-1, 3)
    
    if mask is not None:
        mask = mask.astype(bool).reshape(-1)
        points = points[~mask]

    
    colors = rgb.reshape(-1, 3)
    
    if mask is not None:
        colors = colors[~mask]
    
    # colors between 0 and 1
    colors = colors / 255.0

    # R = cam_ext[:3, :3]
    # t = cam_ext[:3, 3:]

    points /= 1000
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd