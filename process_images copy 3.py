from pathlib import Path
import torch
from torch.utils.data.dataloader import default_collate

import argparse
import os
import cv2
import numpy as np
import open3d as o3d
import copy

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
import pickle
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel
from tools import umeyama_alignment, depth2points
from scipy.optimize import minimize_scalar

'''
python process_images.py --ego_image /local/home/andrye/dev/H2O/subject1/h1/2/cam4/rgb/000043.png --exo_image /local/home/andrye/dev/H2O/subject1/h1/2/cam2/rgb/000043.png --out_folder demo_out
'''

# python process_images.py --images /local/home/andrye/dev/H2O/subject1/h1/2/cam2/rgb/000043.png /local/home/andrye/dev/H2O/subject1/h1/2/cam4/rgb/000043.png --out_folder demo_out --batch_size=1 --side_view --save_mesh --full_frame --body_detector regnety


def initialize(args):
    os.makedirs(args.out_folder, exist_ok=True)

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)


    torch.cuda.empty_cache()
    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    
    return model, model_cfg, device, detector, cpm, renderer


def vit_pose_detection(img, detector, cpm):

    # Detect humans in image
    det_out = detector(img)
    img = img.copy()[:, :, ::-1]

    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
    pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores=det_instances.scores[valid_idx].cpu().numpy()

    # Detect human keypoints for each person
    vitposes_out = cpm.predict_pose(
        img,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes = []
    is_right = []

    for vitposes in vitposes_out:
        left_hand_keyp = vitposes['keypoints'][-42:-21]
        right_hand_keyp = vitposes['keypoints'][-21:]

        # Rejecting not confident detections
        keyp = left_hand_keyp
        valid = keyp[:,2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
            bboxes.append(bbox)
            is_right.append(0)
        keyp = right_hand_keyp
        valid = keyp[:,2] > 0.5
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
    # first hand has to be right hand
    if is_right[0]:
        first = verts[0]
        second = verts[1]
    else:
        first = verts[1]
        second = verts[0]
    return np.concatenate([first, second], axis=0)

def hands_to_mesh(verts, model, mesh_base_color=(1.0, 1.0, 0.9)):
    faces_left = model.mano.faces[:,[0,2,1]]
    faces_right = model.mano.faces
    vertex_colors = np.array([mesh_base_color] * verts.shape[0])
    both_hands_faces = np.concatenate([faces_right, faces_left + verts.shape[0] // 2], axis=0)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(both_hands_faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh

def process_image(img_cv2, model, model_cfg, device, detector, cpm, renderer, out_folder, out_name):
    boxes, right = vit_pose_detection(img_cv2, detector, cpm)
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
    if len(dataset) != 2:
        raise ValueError(f"Dataset length must be 2. Got {len(dataset)}")
    batch = default_collate([dataset[0], dataset[1]])
    batch = recursive_to(batch, device)

    with torch.no_grad():
        out = model(batch)

    multiplier = (2*batch['right']-1)
    pred_cam = out['pred_cam']
    pred_cam[:,1] = multiplier*pred_cam[:,1]
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
    misc_args = dict(
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        focal_length=scaled_focal_length,
        name=os.path.join(out_folder, f'{out_name}_all.obj'),
    )
    cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[0], is_right=all_right, **misc_args)

    # Overlay image
    input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
    input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
    input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

    cv2.imwrite(os.path.join(out_folder, f'{out_name}_all.jpg'), 255*input_img_overlay[:, :, ::-1])
    
    mask = renderer.render_mask_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[0], is_right=all_right, focal_length=scaled_focal_length)
    cv2.imwrite(os.path.join(out_folder, f'{out_name}_all_mask.png'), 255*mask)
    
    return ret_verts, pred_cam, box_center, box_size, img_size, is_right, mask

def compute_alignment_loss(focal_length, ego_pred_cam, ego_box_center, ego_box_size, ego_img_size, ego_ret_verts, ego_is_right,
                           exo_pred_cam, exo_box_center, exo_box_size, exo_img_size, exo_ret_verts, exo_is_right):
    """Computes the Umeyama alignment loss for a given focal length."""
    # Ensure tensors are detached and moved to CPU if necessary
    ego_pred_cam_t = cam_crop_to_full(ego_pred_cam, ego_box_center, ego_box_size, ego_img_size, focal_length).detach().cpu().numpy()
    exo_pred_cam_t = cam_crop_to_full(exo_pred_cam, exo_box_center, exo_box_size, exo_img_size, focal_length).detach().cpu().numpy()
    ego_verts_t = ego_ret_verts.detach().cpu().numpy() + ego_pred_cam_t[:,None,:]
    exo_verts_t = exo_ret_verts.detach().cpu().numpy() + exo_pred_cam_t[:,None,:]
    ego_verts_t = concat_both_hands_verts(ego_verts_t, ego_is_right.cpu().numpy()) # Ensure is_right is numpy
    exo_verts_t = concat_both_hands_verts(exo_verts_t, exo_is_right.cpu().numpy()) # Ensure is_right is numpy
    # Assuming umeyama_alignment returns: R, t, s, loss, transformed_source
    _, _, _, loss, _ = umeyama_alignment(ego_verts_t, exo_verts_t)
    return loss

def rgb_path_to_rest(rgb_path_str):
    """Derives paths for depth, intrinsics, extrinsics, and mask from the RGB image path."""
    rgb_path = Path(rgb_path_str) # Convert the input string to a Path object
    base_dir = rgb_path.parent.parent # Go up from 'rgb' dir to 'camX' dir

    # Use the Path object's attributes and methods
    depth_path = base_dir / 'depth' / rgb_path.name
    cam_int_path = base_dir / 'cam_intrinsics.txt'
    cam_ext_path = base_dir / 'cam_pose' / rgb_path.with_suffix('.txt').name
    return depth_path, cam_int_path, cam_ext_path

def load_from_rgb_path(rgb_path_str):
    depth_path, cam_int_path, cam_ext_path = rgb_path_to_rest(rgb_path_str)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    rgb = cv2.imread(rgb_path_str)
    cam_int = np.loadtxt(cam_int_path)
    cam_ext = np.loadtxt(cam_ext_path).reshape(4, 4)
    return depth, rgb, cam_int, cam_ext


def depth2pcd_masked(depth, rgb, mask, cam_int, cam_ext):
    fx, fy, cx, cy = cam_int[:4]
    # cam_ext = np.loadtxt(cam_ext_path).reshape(4, 4)
    points = depth2points(depth, fx, fy, cx, cy).reshape(-1, 3)
    
    mask = mask.astype(bool).reshape(-1)
    points = points[~mask]

    
    colors = rgb.reshape(-1, 3)
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
            - final_transformation (np.ndarray): The 4x4 transformation matrix from ICP.
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

    # Apply the final transformation to the source point cloud *before* returning
    final_transformation = result_icp.transformation
    source_pcd.transform(final_transformation)

    print("Registration complete.")
    return source_pcd, final_transformation

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--ego_image', type=str, default='', help='Path to ego image')
    parser.add_argument('--exo_image', type=str, default='', help='Path to exo image')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--body_detector', type=str, default='regnety', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    args = parser.parse_args()
    
    model, model_cfg, device, detector, cpm, renderer = initialize(args)
    
    ego_img = cv2.imread(args.ego_image)
    exo_img = cv2.imread(args.exo_image)

    ego_ret_verts, ego_pred_cam, ego_box_center, ego_box_size, ego_img_size, ego_is_right, ego_mask = process_image(ego_img, model, model_cfg, device, detector, cpm, renderer, args.out_folder, 'ego')
    exo_ret_verts, exo_pred_cam, exo_box_center, exo_box_size, exo_img_size, exo_is_right, exo_mask = process_image(exo_img, model, model_cfg, device, detector, cpm, renderer, args.out_folder, 'exo')
    
    # Define bounds for the focal length search (adjust if needed)
    # Avoid 0 as focal length is usually positive.
    focal_length_bounds = (1, 5000)

    # Package arguments for the loss function (excluding focal_length)
    loss_args = (
        ego_pred_cam, ego_box_center, ego_box_size, ego_img_size, ego_ret_verts, ego_is_right,
        exo_pred_cam, exo_box_center, exo_box_size, exo_img_size, exo_ret_verts, exo_is_right
    )

    # Use minimize_scalar to find the optimal focal length
    result = minimize_scalar(
        compute_alignment_loss,
        bounds=focal_length_bounds,
        args=loss_args,
        method='bounded' # Use bounded optimization
    )

    if result.success:
        min_loss_focal_length = result.x
        min_loss = result.fun
        print(f"Optimization successful.")
    else:
        print(f"Optimization failed: {result.message}. Using best found value or default.")
        # Handle optimization failure, e.g., use the found value or a default
        min_loss_focal_length = result.x # Best value found during optimization attempt
        # Recalculate loss just in case result.fun is not reliable on failure
        min_loss = compute_alignment_loss(min_loss_focal_length, *loss_args)
        # Alternatively, consider using a default focal length from model_cfg if available
        # min_loss_focal_length = model_cfg.EXTRA.FOCAL_LENGTH
        # min_loss = compute_alignment_loss(min_loss_focal_length, *loss_args)

    # run the optimal alignment using the found focal length
    ego_pred_cam_t = cam_crop_to_full(ego_pred_cam, ego_box_center, ego_box_size, ego_img_size, min_loss_focal_length).detach().cpu().numpy()
    exo_pred_cam_t = cam_crop_to_full(exo_pred_cam, exo_box_center, exo_box_size, exo_img_size, min_loss_focal_length).detach().cpu().numpy()
    ego_verts_t = ego_ret_verts.detach().cpu().numpy() + ego_pred_cam_t[:,None,:]
    exo_verts_t = exo_ret_verts.detach().cpu().numpy() + exo_pred_cam_t[:,None,:]
    ego_mano_verts = concat_both_hands_verts(ego_verts_t, ego_is_right.cpu().numpy()) # Ensure is_right is numpy
    exo_mano_verts = concat_both_hands_verts(exo_verts_t, exo_is_right.cpu().numpy()) # Ensure is_right is numpy

    # save the meshes as single obj file
    ego_mano_mesh = hands_to_mesh(ego_mano_verts, model)
    exo_mano_mesh = hands_to_mesh(exo_mano_verts, model)
    
    o3d.io.write_triangle_mesh(f"{args.out_folder}/mesh_exo.obj", exo_mano_mesh)
    o3d.io.write_triangle_mesh(f"{args.out_folder}/mesh_ego.obj", ego_mano_mesh)

    # Get the final alignment transformation for reference if needed
    R, t, s, final_loss, transformed_source = umeyama_alignment(ego_mano_verts, exo_mano_verts)
    print(f"Final Alignment Loss (at optimal focal length): {final_loss}") # This should match min_loss

    print(f"Optimal Focal length: {min_loss_focal_length:.2f}")
    
    ego_depth, ego_rgb, ego_cam_int, ego_cam_ext = load_from_rgb_path(args.ego_image)
    exo_depth, exo_rgb, exo_cam_int, exo_cam_ext = load_from_rgb_path(args.exo_image)
    ego_pcd = depth2pcd_masked(ego_depth, ego_rgb, ego_mask, ego_cam_int, ego_cam_ext)
    exo_pcd = depth2pcd_masked(exo_depth, exo_rgb, exo_mask, exo_cam_int, exo_cam_ext)
    
    # uniform sample same number of points as points from mano_mesh
    ego_mano_pcd = ego_mano_mesh.sample_points_uniformly(number_of_points=len(ego_pcd.points))
    exo_mano_pcd = exo_mano_mesh.sample_points_uniformly(number_of_points=len(exo_pcd.points))
    

    # --- Call the registration function for EGO view ---
    print("\n--- Registering EGO View ---")
    # Pass the original sampled points; mean alignment happens inside the function
    aligned_ego_pcd, ego_transformation = register_point_clouds(
        source_pcd=ego_pcd,            # Source: Original sampled MANO points
        target_pcd=ego_mano_pcd,                         # Target: Depth points
        # voxel_size=0.005 # Default is 0.005, uncomment to override
    )
    
    aligned_exo_pcd, exo_transformation = register_point_clouds(
        source_pcd=exo_pcd,            # Source: Original sampled MANO points
        target_pcd=exo_mano_pcd,                         # Target: Depth points
        # voxel_size=0.005 # Default is 0.005, uncomment to override
    )
    
    combined_ego_pcd = ego_mano_pcd + aligned_ego_pcd # Add the original target and the transformed source

    # Save the combined point cloud
    combined_ego_ply_path = f"{args.out_folder}/combined_aligned_ego_pcd.ply"
    o3d.io.write_point_cloud(combined_ego_ply_path, combined_ego_pcd)
    

    combined_exo_pcd = exo_mano_pcd + aligned_exo_pcd # Add the original target and the transformed source
    
    combined_exo_ply_path = f"{args.out_folder}/combined_aligned_exo_pcd.ply"
    o3d.io.write_point_cloud(combined_exo_ply_path, combined_exo_pcd)
    
    print(f"Saved combined EGO point cloud to {combined_ego_ply_path}")
    print(f"Saved combined EXO point cloud to {combined_exo_ply_path}")
    
    
    

if __name__ == '__main__':
    main()
