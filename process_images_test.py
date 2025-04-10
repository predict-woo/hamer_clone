# Standard library imports
import argparse
import os
from pathlib import Path
import copy  # Add import for deepcopy

# Third-party imports
import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.optimize import minimize_scalar
from torch.utils.data.dataloader import default_collate

# Local imports
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel
from tools import umeyama_alignment, depth2pcd, register_point_clouds

# Constants
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
DEFAULT_FOCAL_LENGTH_BOUNDS = (1, 5000)
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_OUTPUT_FOLDER = 'out_demo'

'''
python process_images.py --ego_image /local/home/andrye/dev/H2O/subject1/h1/2/cam4/rgb/000043.png --exo_image /local/home/andrye/dev/H2O/subject1/h1/2/cam2/rgb/000043.png --out_folder demo_out
'''

def initialize(checkpoint, body_detector, out_folder):
    """
    Initialize models, detector, and renderer.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (model, model_cfg, device, detector, cpm, renderer)
    """
    os.makedirs(out_folder, exist_ok=True)

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    torch.cuda.empty_cache()
    
    # Initialize keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    
    return model, model_cfg, device, detector, cpm, renderer


def vit_pose_detection(img, detector, cpm, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Detect human hand keypoints in the image.
    
    Args:
        img (np.ndarray): Input image in BGR format
        detector: Object detector model
        cpm: Keypoint detector model
        confidence_threshold (float): Confidence threshold for keypoint detection
        
    Returns:
        tuple: (bboxes, is_right_hand_flags)
    """
    # Detect humans in image
    det_out = detector(img)
    img_rgb = img.copy()[:, :, ::-1]  # Convert BGR to RGB

    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > confidence_threshold)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    # Detect human keypoints for each person
    vitposes_out = cpm.predict_pose(
        img_rgb,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes = []
    is_right = []

    for vitposes in vitposes_out:
        left_hand_keyp = vitposes['keypoints'][-42:-21]
        right_hand_keyp = vitposes['keypoints'][-21:]

        # Process left hand keypoints
        keyp = left_hand_keyp
        valid = keyp[:,2] > confidence_threshold
        if sum(valid) > 3:
            bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
            bboxes.append(bbox)
            is_right.append(0)
            
        # Process right hand keypoints
        keyp = right_hand_keyp
        valid = keyp[:,2] > confidence_threshold
        if sum(valid) > 3:
            bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
            bboxes.append(bbox)
            is_right.append(1)

    if len(bboxes) == 0:
        raise ValueError("No hands detected")

    boxes = np.stack(bboxes)
    right = np.stack(is_right)

    return boxes, right


def concat_both_hands_verts(verts, is_right):
    """
    Concatenate vertices for both hands with the right hand first.
    
    Args:
        verts (list): List of vertices for each hand
        is_right (list): List of flags indicating if hand is right
        
    Returns:
        np.ndarray: Concatenated vertices with right hand first
    """
    # First hand has to be right hand
    if is_right[0]:
        first = verts[0]
        second = verts[1]
    else:
        first = verts[1]
        second = verts[0]
    return np.concatenate([first, second], axis=0)


def hands_to_mesh(verts, mano_faces, mesh_base_color=(1.0, 1.0, 0.9)):
    """
    Create a mesh from hand vertices.
    
    Args:
        verts (np.ndarray): Hand vertices
        mano_faces (np.ndarray): MANO faces
        mesh_base_color (tuple): RGB color for the mesh
        
    Returns:
        o3d.geometry.TriangleMesh: Triangle mesh of the hands
    """
    faces_left = mano_faces[:,[0,2,1]]
    faces_right = mano_faces
    vertex_colors = np.array([mesh_base_color] * verts.shape[0])
    both_hands_faces = np.concatenate([faces_right, faces_left + verts.shape[0] // 2], axis=0)
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(both_hands_faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh


def process_image(img_cv2, model, model_cfg, device, detector, cpm, renderer, out_folder, out_name):
    """
    Process an image to detect and render hands.
    
    Args:
        img_cv2 (np.ndarray): Input image in BGR format
        model: HaMeR model
        model_cfg: Model configuration
        device: Torch device
        detector: Object detector model
        cpm: Keypoint detector model
        renderer: Renderer object
        out_folder (str): Output folder path
        out_name (str): Output file name prefix
        
    Returns:
        tuple: Hand vertices, camera parameters, and other detection data
    """
    boxes, right = vit_pose_detection(img_cv2, detector, cpm)
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
    
    if len(dataset) != 2:
        raise ValueError(f"Dataset length must be 2. Got {len(dataset)}")
        
    batch = default_collate([dataset[0], dataset[1]])
    batch = recursive_to(batch, device)

    with torch.no_grad():
        out = model(batch)

    # Process model output
    multiplier = (2 * batch['right'] - 1)
    pred_cam = out['pred_cam']
    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    img_size = batch["img_size"].float()
    
    ret_verts = out['pred_vertices']
    ret_verts[:, :, 0] = multiplier.reshape(-1, 1) * ret_verts[:, :, 0]  

    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    
    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
    
    is_right = batch['right']
    
    all_verts = [ret_verts[n].detach().cpu().numpy() for n in range(2)]
    all_cam_t = [pred_cam_t_full[n] for n in range(2)]
    all_right = [is_right[n].cpu().numpy() for n in range(2)]

    # Render front view
    render_args = dict(
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        focal_length=scaled_focal_length,
        name=os.path.join(out_folder, f'{out_name}_all.obj'),
    )
    cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[0], is_right=all_right, **render_args)

    # Overlay image
    input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
    input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2)  # Add alpha channel
    input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

    # Save output images
    cv2.imwrite(os.path.join(out_folder, f'{out_name}_all.jpg'), 255*input_img_overlay[:, :, ::-1])
    
    mask = renderer.render_mask_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[0], is_right=all_right, focal_length=scaled_focal_length)
    cv2.imwrite(os.path.join(out_folder, f'{out_name}_all_mask.png'), 255*mask)
    
    return ret_verts, pred_cam, box_center, box_size, img_size, is_right, mask


def compute_alignment_loss(focal_length, ego_pred_cam, ego_box_center, ego_box_size, ego_img_size, ego_ret_verts, ego_is_right,
                           exo_pred_cam, exo_box_center, exo_box_size, exo_img_size, exo_ret_verts, exo_is_right):
    """
    Computes the Umeyama alignment loss for a given focal length.
    
    Args:
        focal_length (float): Focal length to evaluate
        ego_pred_cam, exo_pred_cam: Camera predictions
        ego/exo_box_center, ego/exo_box_size, ego/exo_img_size: Box parameters
        ego/exo_ret_verts: Hand vertices
        ego/exo_is_right: Hand orientation flags
        
    Returns:
        float: Alignment loss value
    """
    # Convert vertices to point clouds
    ego_pcd = convert_to_global_vertices(
        ego_pred_cam, ego_box_center, ego_box_size, ego_img_size, ego_ret_verts, ego_is_right, 
        focal_length, as_pcd=True
    )
    
    exo_pcd = convert_to_global_vertices(
        exo_pred_cam, exo_box_center, exo_box_size, exo_img_size, exo_ret_verts, exo_is_right, 
        focal_length, as_pcd=True
    )
    
    # Calculate alignment loss
    _, loss, _ = umeyama_alignment(ego_pcd, exo_pcd)
    return loss


def rgb_path_to_rest(rgb_path_str):
    """
    Derives paths for depth, intrinsics, and extrinsics from the RGB image path.
    
    Args:
        rgb_path_str (str): Path to an RGB image
        
    Returns:
        tuple: (depth_path, cam_int_path, cam_ext_path)
    """
    rgb_path = Path(rgb_path_str)
    base_dir = rgb_path.parent.parent  # Go up from 'rgb' dir to 'camX' dir

    depth_path = base_dir / 'depth' / rgb_path.name
    cam_int_path = base_dir / 'cam_intrinsics.txt'
    cam_ext_path = base_dir / 'cam_pose' / rgb_path.with_suffix('.txt').name
    return depth_path, cam_int_path, cam_ext_path


def load_from_rgb_path(rgb_path_str):
    """
    Load depth, RGB, camera intrinsics and extrinsics from an RGB path.
    
    Args:
        rgb_path_str (str): Path to an RGB image
        
    Returns:
        tuple: (depth, rgb, cam_int, cam_ext)
    """
    depth_path, cam_int_path, cam_ext_path = rgb_path_to_rest(rgb_path_str)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    rgb = cv2.imread(rgb_path_str)
    cam_int = np.loadtxt(str(cam_int_path))
    cam_ext = np.loadtxt(str(cam_ext_path)).reshape(4, 4)
    return depth, rgb, cam_int, cam_ext


def convert_to_global_vertices(pred_cam, box_center, box_size, img_size, ret_verts, is_right, focal_length, as_pcd=False):
    """
    Convert predicted camera parameters and vertices to global space and combine both hands.
    
    Args:
        pred_cam (torch.Tensor): Camera prediction tensor
        box_center (torch.Tensor): Box center coordinates
        box_size (torch.Tensor): Box size
        img_size (torch.Tensor): Image dimensions
        ret_verts (torch.Tensor): Predicted vertices
        is_right (torch.Tensor): Hand orientation indicators (1 for right, 0 for left)
        focal_length (float): Focal length for camera conversion
        as_pcd (bool): Whether to return as Open3D point cloud
        
    Returns:
        Either numpy array of vertices or Open3D point cloud
    """
    # Convert camera parameters to full image coordinates
    pred_cam_t = cam_crop_to_full(pred_cam, box_center, box_size, img_size, focal_length).detach().cpu().numpy()
    
    # Transform vertices to global space
    verts_t = ret_verts.detach().cpu().numpy() + pred_cam_t[:,None,:]
    
    # Combine both hands ensuring right hand comes first
    mano_verts = concat_both_hands_verts(verts_t, is_right.cpu().numpy())
    
    # Optionally convert to point cloud
    if as_pcd:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mano_verts)
        return pcd
    
    return mano_verts


def optimize_focal_length(ego_params, exo_params):
    """
    Find the optimal focal length that minimizes alignment error.
    
    Args:
        ego_params (tuple): Parameters for ego view
        exo_params (tuple): Parameters for exo view
        
    Returns:
        tuple: (optimal_focal_length, min_loss)
    """
    # Define bounds for the focal length search
    focal_length_bounds = DEFAULT_FOCAL_LENGTH_BOUNDS

    # Package all parameters together
    loss_args = ego_params + exo_params

    # Find the optimal focal length
    result = minimize_scalar(
        compute_alignment_loss,
        bounds=focal_length_bounds,
        args=loss_args,
        method='bounded'
    )

    if result.success:
        optimal_focal_length = result.x
        min_loss = result.fun
        print(f"Optimization successful.")
    else:
        print(f"Optimization failed: {result.message}. Using best found value.")
        optimal_focal_length = result.x
        min_loss = compute_alignment_loss(optimal_focal_length, *loss_args)

    return optimal_focal_length, min_loss


def save_aligned_point_clouds(ego_mesh, exo_mesh, ego_pcd, exo_pcd, alignment_transform, out_folder):
    """
    Save the aligned point clouds.
    
    Args:
        args: Command line arguments
        ego_mesh, exo_mesh: Hand meshes
        ego_pcd, exo_pcd: Point clouds
        alignment_transform: Transformation matrix
    """
    # Create uniformly sampled point clouds from meshes
    ego_mano_pcd = ego_mesh.sample_points_uniformly(number_of_points=len(ego_pcd.points))
    exo_mano_pcd = exo_mesh.sample_points_uniformly(number_of_points=len(exo_pcd.points))

    # Register ego point cloud
    print("\n--- Registering EGO View ---")
    aligned_ego_mano_pcd, ego_transformation = register_point_clouds(
        source_pcd=ego_mano_pcd,
        target_pcd=ego_pcd
    )
    
    # Register exo point cloud
    print("\n--- Registering EXO View ---")
    aligned_exo_mano_pcd, exo_transformation = register_point_clouds(
        source_pcd=exo_mano_pcd,
        target_pcd=exo_pcd
    )
    
    # Combine and save point clouds
    combined_ego_pcd = ego_pcd + aligned_ego_mano_pcd
    combined_ego_ply_path = f"{out_folder}/combined_aligned_ego_pcd.ply"
    o3d.io.write_point_cloud(combined_ego_ply_path, combined_ego_pcd)
    
    combined_exo_pcd = exo_pcd + aligned_exo_mano_pcd
    combined_exo_ply_path = f"{out_folder}/combined_aligned_exo_pcd.ply"
    o3d.io.write_point_cloud(combined_exo_ply_path, combined_exo_pcd)
    
    print(f"Saved combined EGO point cloud to {combined_ego_ply_path}")
    print(f"Saved combined EXO point cloud to {combined_exo_ply_path}")

    # Check transformation chains
    check_mano_transform = copy.deepcopy(ego_mano_pcd)
    check_mano_transform.transform(alignment_transform)
    check_mano_transform += exo_mano_pcd
    o3d.io.write_point_cloud(f"{out_folder}/check_mano_transform.ply", check_mano_transform)

    check_ego_transform = copy.deepcopy(ego_mano_pcd)
    check_ego_transform.transform(ego_transformation)
    check_ego_transform += ego_pcd
    o3d.io.write_point_cloud(f"{out_folder}/check_ego_transform.ply", check_ego_transform)
    
    check_exo_transform = copy.deepcopy(exo_mano_pcd)
    check_exo_transform.transform(exo_transformation)
    check_exo_transform += exo_pcd
    o3d.io.write_point_cloud(f"{out_folder}/check_exo_transform.ply", check_exo_transform)

    # Calculate the transformation chain
    ego_transformation_inv = np.linalg.inv(ego_transformation)
    transformation_chain = np.matmul(exo_transformation, np.matmul(alignment_transform, ego_transformation_inv))
    
    # Apply the transformation chain to ego_pcd
    transformed_ego_pcd = copy.deepcopy(ego_pcd)
    transformed_ego_pcd.transform(transformation_chain)
    
    # Visualize and save the result
    combined_transformed_pcd = transformed_ego_pcd + exo_pcd
    o3d.io.write_point_cloud(f"{out_folder}/ego_to_exo_transformed.ply", combined_transformed_pcd)
    
    return transformation_chain



def full_pipeline(checkpoint, body_detector, ego_image, exo_image, out_folder):

    # Initialize models
    model, model_cfg, device, detector, cpm, renderer = initialize(checkpoint, body_detector, out_folder)
    
    # Load and process images
    ego_img = cv2.imread(ego_image)
    exo_img = cv2.imread(exo_image)

    ego_ret_verts, ego_pred_cam, ego_box_center, ego_box_size, ego_img_size, ego_is_right, ego_mask = process_image(
        ego_img, model, model_cfg, device, detector, cpm, renderer, out_folder, 'ego'
    )
    
    exo_ret_verts, exo_pred_cam, exo_box_center, exo_box_size, exo_img_size, exo_is_right, exo_mask = process_image(
        exo_img, model, model_cfg, device, detector, cpm, renderer, out_folder, 'exo'
    )
    
    # Package parameters for focal length optimization
    ego_params = (
        ego_pred_cam, ego_box_center, ego_box_size, ego_img_size, ego_ret_verts, ego_is_right
    )
    
    exo_params = (
        exo_pred_cam, exo_box_center, exo_box_size, exo_img_size, exo_ret_verts, exo_is_right
    )
    
    # Find optimal focal length
    optimal_focal_length, min_loss = optimize_focal_length(ego_params, exo_params)
    print(f"Optimal Focal length: {optimal_focal_length:.2f}")
    print(f"Final Alignment Loss: {min_loss}")

    # Generate meshes with optimal focal length
    ego_mano_verts = convert_to_global_vertices(
        ego_pred_cam, ego_box_center, ego_box_size, ego_img_size, ego_ret_verts, ego_is_right, 
        optimal_focal_length
    )
    
    exo_mano_verts = convert_to_global_vertices(
        exo_pred_cam, exo_box_center, exo_box_size, exo_img_size, exo_ret_verts, exo_is_right, 
        optimal_focal_length
    )

    # Create meshes from the vertices
    ego_mano_mesh = hands_to_mesh(ego_mano_verts, model.mano.faces)
    exo_mano_mesh = hands_to_mesh(exo_mano_verts, model.mano.faces)
    
    # Save meshes
    o3d.io.write_triangle_mesh(f"{out_folder}/mesh_exo.obj", exo_mano_mesh)
    o3d.io.write_triangle_mesh(f"{out_folder}/mesh_ego.obj", ego_mano_mesh)

    # Create point clouds for alignment
    ego_mano_pcd = o3d.geometry.PointCloud()
    ego_mano_pcd.points = o3d.utility.Vector3dVector(ego_mano_verts)
    
    exo_mano_pcd = o3d.geometry.PointCloud()
    exo_mano_pcd.points = o3d.utility.Vector3dVector(exo_mano_verts)

    # Get the final alignment transformation and transformed point cloud
    alignment_transform, final_loss, transformed_ego_pcd = umeyama_alignment(ego_mano_pcd, exo_mano_pcd)
    
    # Load depth data and create point clouds
    ego_depth, ego_rgb, ego_cam_int, ego_cam_ext = load_from_rgb_path(ego_image)
    exo_depth, exo_rgb, exo_cam_int, exo_cam_ext = load_from_rgb_path(exo_image)
    ego_pcd = depth2pcd(ego_depth, ego_rgb, ego_cam_int, ego_cam_ext, mask=ego_mask)
    exo_pcd = depth2pcd(exo_depth, exo_rgb, exo_cam_int, exo_cam_ext, mask=exo_mask)
    
    # Save aligned point clouds and get transformation chain
    transformation_chain = save_aligned_point_clouds(
        ego_mano_mesh, exo_mano_mesh, ego_pcd, exo_pcd, alignment_transform, out_folder
    )

    # Generate and save full point clouds
    ego_total_pcd = depth2pcd(ego_depth, ego_rgb, ego_cam_int, ego_cam_ext)
    exo_total_pcd = depth2pcd(exo_depth, exo_rgb, exo_cam_int, exo_cam_ext)

    # Transform and combine ego point cloud to exo frame
    transformed_ego_total_pcd = copy.deepcopy(ego_total_pcd)
    transformed_ego_total_pcd.transform(transformation_chain)
    combined_total_pcd = transformed_ego_total_pcd + exo_total_pcd
    o3d.io.write_point_cloud(f"{out_folder}/combined_pcd.ply", combined_total_pcd)
    print(f"Saved combined point cloud to {out_folder}/combined_pcd.ply")


def main():
    """Main function to process images and align hand models."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--ego_image', type=str, required=True, help='Path to ego image')
    parser.add_argument('--exo_image', type=str, required=True, help='Path to exo image')
    parser.add_argument('--out_folder', type=str, default=DEFAULT_OUTPUT_FOLDER, help='Output folder to save rendered results')
    parser.add_argument('--body_detector', type=str, default='regnety', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    args = parser.parse_args()
    
    full_pipeline(args.checkpoint, args.body_detector, args.ego_image, args.exo_image, args.out_folder)




if __name__ == '__main__':
    main()
