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

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

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
print("faces loaded")

# load the output
out = np.load('demo_out/ego.png_out.npz')
print("out loaded")

# load the batch
batch = np.load('demo_out/ego.png_batch.npz')
print("batch loaded")

batch = np_dict2torch_dict(batch)
out = np_dict2torch_dict(out)

multiplier = (2*batch['right']-1)
pred_cam = out['pred_cam']
pred_cam[:,1] = multiplier*pred_cam[:,1]
box_center = batch["box_center"].float()
box_size = batch["box_size"].float()
img_size = batch["img_size"].float()
multiplier = (2*batch['right']-1)
scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

batch_size = batch['img'].shape[0]

renderer = Renderer(model_cfg, faces)

for n in range(batch_size):
    person_id = int(batch['personid'][n])
    white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
    input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
    input_patch = input_patch.permute(1,2,0).numpy()
    
    # render pred_vertices from camera perspective using open3d
    pred_vertices = out['pred_vertices'][n]
    pred_cam_t = out['pred_cam_t'][n]
    
    # render_mano(pred_vertices.detach().cpu().numpy(), pred_cam_t.detach().cpu().numpy(), mesh_base_color=LIGHT_BLUE)
    

    regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                            out['pred_cam_t'][n].detach().cpu().numpy(),
                            batch['img'][n],
                            mesh_base_color=LIGHT_BLUE,
                            scene_bg_color=(1, 1, 1),
                            )
    
    # save the image
    cv2.imwrite(f'regression_img_{n}.png', regression_img)

    


# pred_keypoints_3d = out['pred_keypoints_3d']

# keypoint_left = pred_keypoints_3d[0]
# keypoint_right = pred_keypoints_3d[1]

# render the keypoints