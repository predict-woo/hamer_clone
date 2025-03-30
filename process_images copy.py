from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
import pickle
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--images', nargs='+', type=str, default='images', help='list of paths to images')
    parser.add_argument('--output_names', nargs='+', type=str, default='', help='list of output names')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='regnety', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')

    args = parser.parse_args()

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

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)
    # Get all demo images ends with .jpg or .png
    img_paths = [img for img in args.images]
    
    if len(args.output_names) != len(img_paths):
        raise ValueError(f"Number of output names must be equal to number of images. Got {len(args.output_names)} and {len(img_paths)}")
    # Iterate over all images in folder
    for img_path, output_name in zip(img_paths, args.output_names):
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

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

        # Use hands based on hand keypoint detections
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
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            
            # if batch is not length 2, raise error
            if len(batch) != 2:
                raise ValueError(f"Batch length must be 2. Got {len(batch)}")
            
            # save batch as pickle
            save_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    save_batch[key] = value.detach().cpu().numpy()
                else:
                    # print key and type of value
                    print(key, type(value))
            
            np.savez(os.path.join(args.out_folder, f'{output_name}_batch.npz'), **save_batch)
            
            with torch.no_grad():
                out = model(batch)
            
            # use pickle to save out, a dictionary of tensors
            save_out = {}
            for key, value in out.items():
                if isinstance(value, torch.Tensor):
                    save_out[key] = value.detach().cpu().numpy()
                else:
                    # print key and type of value
                    print(key, type(value))
            
            # save as pickle

            # # Save the dictionary as a .npz file
            np.savez(os.path.join(args.out_folder, f'{output_name}_out.npz'), **save_out)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
            
            
            
            
            np.save(os.path.join(args.out_folder, f'{output_name}_pred_cam.npy'), out['pred_cam'].detach().cpu().numpy())
            np.save(os.path.join(args.out_folder, f'{output_name}_pred_cam_t.npy'), out['pred_cam_t'].detach().cpu().numpy())
            np.save(os.path.join(args.out_folder, f'{output_name}_focal_length.npy'), out['focal_length'].detach().cpu().numpy())
            
            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()
                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

        # Render front view
        misc_args = dict(
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length,
            name=os.path.join(args.out_folder, f'{output_name}_all.obj'),
        )
        cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

        # Overlay image
        input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
        input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
        input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

        cv2.imwrite(os.path.join(args.out_folder, f'{output_name}_all.jpg'), 255*input_img_overlay[:, :, ::-1])
        
        mask = renderer.render_mask_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, focal_length=scaled_focal_length)
        cv2.imwrite(os.path.join(args.out_folder, f'{output_name}_all_mask.png'), 255*mask)

if __name__ == '__main__':
    main()
