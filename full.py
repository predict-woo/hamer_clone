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


import json
from typing import Dict, Optional
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import hamer

from exo2ego import *

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--image', type=str, default='/cluster/project/cvg/data/H2O/subject4/h2/3/cam2/rgb/000022.png', help='input image')
    parser.add_argument('--depth', type=str, default='/cluster/project/cvg/data/H2O/subject4/h2/3/cam2/depth/000022.png', help='input depth')
    parser.add_argument('--output', type=str, default='output', help='output folder')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')

    args = parser.parse_args()

    # 1. Download HaMeR and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # 2. Load detector
    cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # 3. keypoint detector
    cpm = ViTPoseModel(device)

    # 4. Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # 5. Make output directory if it does not exist
    os.makedirs(args.output, exist_ok=True)
    
    # 6. Read image and depth
    img_path = args.image
    image_name = os.path.basename(img_path)
    img_cv2 = cv2.imread(str(img_path))
    depth_path = args.depth
    depth_cv2 = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    
    print(depth_cv2)

    # 7. Detect humans in image
    det_out = detector(img_cv2)
    img = img_cv2.copy()[:, :, ::-1]

    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
    pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores=det_instances.scores[valid_idx].cpu().numpy()

    # 8. Detect human keypoints for each person
    vitposes_out = cpm.predict_pose(
        img,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes = []
    is_right = []

    # 9. Use hands based on hand keypoint detections
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
        print("No humans detected")
        return

    boxes = np.stack(bboxes)
    right = np.stack(is_right)

    # Run reconstruction on all detected hands
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
    
    print(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    all_verts = []
    all_cam_t = []
    all_right = []
    
    for batch in dataloader:
        print(batch)
        batch = recursive_to(batch, device)
        
        # save batch as pickle
        save_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                save_batch[key] = value.detach().cpu().numpy()
            else:
                # print key and type of value
                print(key, type(value))
                
        with torch.no_grad():
            out = model(batch)
        
        multiplier = (2*batch['right']-1)
        pred_cam = out['pred_cam']
        pred_cam[:,1] = multiplier*pred_cam[:,1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
        
        
        
        verts = out['pred_vertices']
        is_right = batch['right']
        verts[:, :, 0] = multiplier.reshape(-1, 1)  * verts[:, :, 0]  
        

        verts = verts + pred_cam_t_full[:,None,:]
        
        
        

if __name__ == '__main__':
    main()
