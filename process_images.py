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
from tools import umeyama_alignment, depth2points, register_point_clouds, depth2pcd
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

def hands_to_mesh(verts, mano_faces, mesh_base_color=(1.0, 1.0, 0.9)):
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
    # Convert vertices using the helper function
    ego_pcd = convert_to_global_vertices(
        ego_pred_cam, ego_box_center, ego_box_size, ego_img_size, ego_ret_verts, ego_is_right, 
        focal_length, as_pcd=True
    )
    
    exo_pcd = convert_to_global_vertices(
        exo_pred_cam, exo_box_center, exo_box_size, exo_img_size, exo_ret_verts, exo_is_right, 
        focal_length, as_pcd=True
    )
    
    # Use the updated umeyama_alignment function that works with point clouds
    _, loss, _ = umeyama_alignment(ego_pcd, exo_pcd)
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

def convert_to_global_vertices(pred_cam, box_center, box_size, img_size, ret_verts, is_right, focal_length, as_pcd=False):
    """
    Convert predicted camera parameters and vertices to global space and combine both hands.
    
    Args:
        pred_cam (torch.Tensor): Camera prediction tensor.
        box_center (torch.Tensor): Box center coordinates.
        box_size (torch.Tensor): Box size.
        img_size (torch.Tensor): Image dimensions.
        ret_verts (torch.Tensor): Predicted vertices.
        is_right (torch.Tensor): Hand orientation indicators (1 for right, 0 for left).
        focal_length (float): Focal length for camera conversion.
        as_pcd (bool, optional): Whether to return as Open3D point cloud. Default: False.
        
    Returns:
        Either numpy array of vertices or Open3D point cloud, depending on as_pcd parameter.
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
    
    # Define bounds for the focal length search
    focal_length_bounds = (1, 5000)

    # Package arguments for the loss function
    loss_args = (
        ego_pred_cam, ego_box_center, ego_box_size, ego_img_size, ego_ret_verts, ego_is_right,
        exo_pred_cam, exo_box_center, exo_box_size, exo_img_size, exo_ret_verts, exo_is_right
    )

    # Find the optimal focal length
    result = minimize_scalar(
        compute_alignment_loss,
        bounds=focal_length_bounds,
        args=loss_args,
        method='bounded'
    )

    if result.success:
        min_loss_focal_length = result.x
        min_loss = result.fun
        print(f"Optimization successful.")
    else:
        print(f"Optimization failed: {result.message}. Using best found value or default.")
        min_loss_focal_length = result.x
        min_loss = compute_alignment_loss(min_loss_focal_length, *loss_args)

    # Run the optimal alignment using the found focal length
    ego_mano_verts = convert_to_global_vertices(
        ego_pred_cam, ego_box_center, ego_box_size, ego_img_size, ego_ret_verts, ego_is_right, 
        min_loss_focal_length
    )
    exo_mano_verts = convert_to_global_vertices(
        exo_pred_cam, exo_box_center, exo_box_size, exo_img_size, exo_ret_verts, exo_is_right, 
        min_loss_focal_length
    )

    # Create meshes from the vertices
    ego_mano_mesh = hands_to_mesh(ego_mano_verts, model.mano.faces)
    exo_mano_mesh = hands_to_mesh(exo_mano_verts, model.mano.faces)
    
    o3d.io.write_triangle_mesh(f"{args.out_folder}/mesh_exo.obj", exo_mano_mesh)
    o3d.io.write_triangle_mesh(f"{args.out_folder}/mesh_ego.obj", ego_mano_mesh)

    # Create point clouds for alignment
    ego_mano_pcd = o3d.geometry.PointCloud()
    ego_mano_pcd.points = o3d.utility.Vector3dVector(ego_mano_verts)
    
    exo_mano_pcd = o3d.geometry.PointCloud()
    exo_mano_pcd.points = o3d.utility.Vector3dVector(exo_mano_verts)

    # Get the final alignment transformation and transformed point cloud
    alignment_transform, final_loss, transformed_ego_pcd = umeyama_alignment(ego_mano_pcd, exo_mano_pcd)
    print(f"Final Alignment Loss (at optimal focal length): {final_loss}")

    print(f"Optimal Focal length: {min_loss_focal_length:.2f}")
    
    # Load depth data and create point clouds
    ego_depth, ego_rgb, ego_cam_int, ego_cam_ext = load_from_rgb_path(args.ego_image)
    exo_depth, exo_rgb, exo_cam_int, exo_cam_ext = load_from_rgb_path(args.exo_image)
    ego_pcd = depth2pcd(ego_depth, ego_rgb, ego_cam_int, ego_cam_ext, mask=ego_mask)
    exo_pcd = depth2pcd(exo_depth, exo_rgb, exo_cam_int, exo_cam_ext, mask=exo_mask)
    
    # Uniform sample same number of points from MANO meshes
    ego_mano_pcd = ego_mano_mesh.sample_points_uniformly(number_of_points=len(ego_pcd.points))
    exo_mano_pcd = exo_mano_mesh.sample_points_uniformly(number_of_points=len(exo_pcd.points))

    # Register point clouds
    print("\n--- Registering EGO View ---")
    aligned_ego_mano_pcd, ego_transformation = register_point_clouds(
        source_pcd=ego_mano_pcd,
        target_pcd=ego_pcd
    )
    
    print("\n--- Registering EXO View ---")
    aligned_exo_mano_pcd, exo_transformation = register_point_clouds(
        source_pcd=exo_mano_pcd,
        target_pcd=exo_pcd
    )
    
    # Combine and save point clouds
    combined_ego_pcd = ego_pcd + aligned_ego_mano_pcd
    combined_ego_ply_path = f"{args.out_folder}/combined_aligned_ego_pcd.ply"
    o3d.io.write_point_cloud(combined_ego_ply_path, combined_ego_pcd)
    
    combined_exo_pcd = exo_pcd + aligned_exo_mano_pcd
    combined_exo_ply_path = f"{args.out_folder}/combined_aligned_exo_pcd.ply"
    o3d.io.write_point_cloud(combined_exo_ply_path, combined_exo_pcd)
    
    print(f"Saved combined EGO point cloud to {combined_ego_ply_path}")
    print(f"Saved combined EXO point cloud to {combined_exo_ply_path}")


    # check each transformation chain
    check_mano_transform = copy.deepcopy(ego_mano_pcd).transform(alignment_transform) + copy.deepcopy(exo_mano_pcd)
    o3d.io.write_point_cloud(f"{args.out_folder}/check_mano_transform.ply", check_mano_transform)


    check_ego_transform = copy.deepcopy(ego_mano_pcd).transform(ego_transformation) + copy.deepcopy(ego_pcd)
    o3d.io.write_point_cloud(f"{args.out_folder}/check_ego_transform.ply", check_ego_transform)
    
    check_exo_transform = copy.deepcopy(exo_mano_pcd).transform(exo_transformation) + copy.deepcopy(exo_pcd)
    o3d.io.write_point_cloud(f"{args.out_folder}/check_exo_transform.ply", check_exo_transform)


    # Calculate the transformation chain
    # Step 1: Get the inverse of ego_transformation
    ego_transformation_inv = np.linalg.inv(ego_transformation)
    
    # Step 2: Compute the complete transformation chain
    # T_chain = T₃ × T₂ × T₁ = exo_transformation × alignment_transform × inv(ego_transformation)
    transformation_chain = np.matmul(exo_transformation, np.matmul(alignment_transform, ego_transformation_inv))
    
    # Step 3: Apply the transformation chain to ego_pcd
    transformed_ego_pcd = copy.deepcopy(ego_pcd).transform(transformation_chain)
    
    # Visualize and save the result
    combined_transformed_pcd = transformed_ego_pcd + exo_pcd
    o3d.io.write_point_cloud(f"{args.out_folder}/ego_to_exo_transformed.ply", combined_transformed_pcd)
    
    print(f"Transformation chain computed and applied.")
    print(f"Saved transformed point cloud to {args.out_folder}/ego_to_exo_transformed.ply")

    ego_total_pcd = depth2pcd(ego_depth, ego_rgb, ego_cam_int, ego_cam_ext)
    exo_total_pcd = depth2pcd(exo_depth, exo_rgb, exo_cam_int, exo_cam_ext)

    # Uniform sample same number of points from MANO meshes
    transformed_ego_pcd = ego_total_pcd.transform(transformation_chain)
    
    combined_pcd = transformed_ego_pcd + exo_total_pcd
    o3d.io.write_point_cloud(f"{args.out_folder}/combined_pcd.ply", combined_pcd)

if __name__ == '__main__':
    main()
