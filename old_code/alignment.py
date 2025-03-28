import numpy as np
from scipy.spatial import procrustes
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import os
import torch

def load_obj_vertices(obj_path):
    """Load vertices from an OBJ file."""
    mesh = trimesh.load(obj_path)
    return np.array(mesh.vertices), mesh


def load_camera_pose(file_path):
    """Load camera pose matrix from a text file."""
    with open(file_path, 'r') as f:
        line = f.readline().strip()
        # Parse the 16 values and reshape into a 4x4 matrix
        values = [float(val) for val in line.split()]
        if len(values) != 16:
            raise ValueError(f"Expected 16 values for camera pose, got {len(values)}")
        return np.array(values).reshape(4, 4)


camera_pose_exo = load_camera_pose("/cluster/scratch/andrye/h2o/subject1/h1/0/cam0/cam_pose/000001.txt")
camera_pose_ego = load_camera_pose("/cluster/scratch/andrye/h2o/subject1/h1/0/cam4/cam_pose/000001.txt")

ego_pred_cam = np.load("demo_out/ego_pred_cam.npy")
ego_pred_cam_t = np.load("demo_out/ego_pred_cam_t.npy")
ego_focal_length = np.load("demo_out/ego_focal_length.npy")

exo_pred_cam = np.load("demo_out/exo_pred_cam.npy")
exo_pred_cam_t = np.load("demo_out/exo_pred_cam_t.npy")
exo_focal_length = np.load("demo_out/exo_focal_length.npy")

ego_vertices, ego_mesh = load_obj_vertices("demo_out/ego_1.obj")
exo_vertices, exo_mesh = load_obj_vertices("demo_out/exo_0.obj")

def umeyama_alignment(source, target):
    """
    Umeyama algorithm to find rotation, translation and scaling between two point sets.
    
    Parameters:
    source: (N, D) array of source points
    target: (N, D) array of target points
    
    Returns:
    R: (D, D) rotation matrix
    t: (D,) translation vector
    s: scaling factor
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
    u, s, vh = np.linalg.svd(cov)
    
    # Handle reflection case
    reflection_matrix = np.eye(d)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        reflection_matrix[d-1, d-1] = -1
    
    # Calculate rotation
    R = u @ reflection_matrix @ vh
    
    # Calculate scaling
    var_source = np.sum(np.var(source_centered, axis=0))
    s = 1.0 if var_source == 0 else np.sum(s) / var_source
    
    # Calculate translation
    t = target_centroid - s * R @ source_centroid
    
    return R, t, s

def plot_mesh(ax, vertices, faces, color, alpha=0.7, title=None):
    """Plot a mesh on the given axis."""
    # Plot the mesh as a collection of triangles
    tri = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                         triangles=faces, color=color, alpha=alpha, shade=True)
    
    if title:
        ax.set_title(title)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    return tri


def get_predicted_camera_matrix(pred_cam, pred_cam_t, focal_length):
    """
    Convert the predicted camera parameters to a 4x4 transformation matrix.
    
    Args:
        pred_cam: Camera rotation/orientation parameters
        pred_cam_t: Camera translation parameters
        focal_length: Focal length of the camera
        
    Returns:
        4x4 camera transformation matrix
    """
    # Convert parameters to numpy if they're tensors
    pred_cam_t = pred_cam_t[0]

    # Create rotation matrix from pred_cam
    rotation = np.eye(3)  # Default to identity rotation
    # Create translation vector
    translation = pred_cam_t.reshape(3)
    
    # Construct 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    
    return transform

def calculate_camera_pose_error(gt_pose, pred_pose):
    """
    Calculate error between ground truth and predicted camera poses.
    
    Args:
        gt_pose: Ground truth 4x4 camera pose matrix
        pred_pose: Predicted 4x4 camera pose matrix
        
    Returns:
        Dictionary with different error metrics
    """
    # Extract rotation matrices
    gt_R = gt_pose[:3, :3]
    pred_R = pred_pose[:3, :3]
    
    # Extract translation vectors
    gt_t = gt_pose[:3, 3]
    pred_t = pred_pose[:3, 3]
    
    # Calculate rotation error (Frobenius norm of difference)
    R_error = np.linalg.norm(gt_R - pred_R, 'fro')
    
    # Calculate translation error (Euclidean distance)
    t_error = np.linalg.norm(gt_t - pred_t)
    
    # Calculate overall pose error (Frobenius norm of difference of full matrices)
    pose_error = np.linalg.norm(gt_pose - pred_pose, 'fro')
    
    return {
        'rotation_error': R_error,
        'translation_error': t_error,
        'pose_error': pose_error
    }

def main():
    
    # Find transformation
    R, t, s = umeyama_alignment(ego_vertices, exo_vertices)
    conversion_matrix = np.eye(4)
    conversion_matrix[:3, :3] = R
    conversion_matrix[:3, 3] = t
    
    # Set numpy print options for consistent formatting
    np.set_printoptions(suppress=True, precision=8, floatmode='fixed')
    
    print("Rotation matrix:")
    print(R)
    print("\nTranslation vector:")
    print(t)
    print("\nScaling factor:")
    print(s)
    
    # Apply transformation to all vertices
    transformed_ego_vertices = s * ego_vertices @ R.T + t
    
    # Create a new mesh with transformed vertices
    transformed_ego_mesh = trimesh.Trimesh(vertices=transformed_ego_vertices, 
                                          faces=ego_mesh.faces)
    
    # Calculate error
    transformed_ego_subset = s * ego_vertices @ R.T + t
    error = np.mean(np.linalg.norm(transformed_ego_subset - exo_vertices, axis=1))
    print(f"\nMean alignment error: {error:.8f}")
    
    # Visualization code (unchanged)
    # Create a figure with 4 subplots in a 2x2 grid
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot original ego mesh
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    plot_mesh(ax1, ego_vertices, ego_mesh.faces, 'blue', title='Original Ego Mesh')
    
    # Plot original exo mesh
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    plot_mesh(ax2, exo_vertices, exo_mesh.faces, 'red', title='Original Exo Mesh')
    
    # Plot transformed ego mesh
    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    plot_mesh(ax3, transformed_ego_vertices, ego_mesh.faces, 'green', 
             title='Transformed Ego Mesh')
    
    # Plot comparison of transformed ego and original exo
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    plot_mesh(ax4, transformed_ego_vertices, ego_mesh.faces, 'green', alpha=0.5, 
             title='Overlay: Transformed Ego (green) vs Exo (red)')
    plot_mesh(ax4, exo_vertices, exo_mesh.faces, 'red', alpha=0.5)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('demo_out', exist_ok=True)
    
    # Save the combined figure
    plt.savefig('alignment_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved comparison image to 'alignment_comparison.png'")
    
    np.set_printoptions()

if __name__ == "__main__":
    main()


