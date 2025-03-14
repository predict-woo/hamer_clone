import numpy as np
import pickle
import open3d as o3d
# demo_out/ego.png_out.npz

from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
import torch
import cv2

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

# model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)

# # save model_cfg to file
# with open('model_cfg.pkl', 'wb') as f:
#     pickle.dump(model_cfg, f)

# # save model.mano.faces to file
# faces = model.mano.faces
# if torch.is_tensor(faces) and faces.is_cuda:
#     faces = faces.cpu().numpy()
# with open('model_mano_faces.pkl', 'wb') as f:
#     pickle.dump(faces, f)

# # image from example_data/ego.png
# img = cv2.imread('example_data/ego.png')
# # convert img to torch tensor
# img = torch.from_numpy(img)

# pred_keypoints_3d = data['pred_keypoints_3d']

# keypoint_left = pred_keypoints_3d[0]
# keypoint_right = pred_keypoints_3d[1]


# res_img = renderer(data['pred_vertices'][0],
#          data['pred_cam_t'][0],
#          img,
#          mesh_base_color=LIGHT_BLUE,
#          scene_bg_color=(1, 1, 1),
#          )

# cv2.imwrite('pred_vertices.png', res_img)


# model_mano_faces = pickle.load(open('model_mano_faces.pkl', 'rb'))
# model_cfg = pickle.load(open('model_cfg.pkl', 'rb'))

# renderer = Renderer(model_cfg, faces=model_mano_faces)

# data = pickle.load(open('demo_out/ego.png_out.pkl', 'rb'))

# model_cfg = pickle.load(open('model_cfg.pkl', 'rb'))
# out = np.load('demo_out/ego.png_out.npz')

# multiplier = (2*batch['right']-1)
# pred_cam = out['pred_cam']
# pred_cam[:,1] = multiplier*pred_cam[:,1]
# box_center = batch["box_center"].float()
# box_size = batch["box_size"].float()
# img_size = batch["img_size"].float()
# multiplier = (2*batch['right']-1)
# scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()



# keypoint_left = pred_keypoints_3d[0]
# keypoint_right = pred_keypoints_3d[1]

# print(keypoint_left.shape)
# print(keypoint_right.shape)






# print(keypoint_left)

# render the keypoints 

# # save as ply file
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(keypoint_left)
# o3d.io.write_point_cloud("keypoint_left.ply", point_cloud)





# crop_img = ((data['crop_img_np'] * np.array([0.229, 0.224, 0.225]).reshape(3,1,1)) + np.array([0.485, 0.456, 0.406]).reshape(3,1,1)).transpose((1,2,0))
# is_right = np.array([1], dtype=np.float32)

# verts = vertices
# verts[:, 0] = (2 * is_right -1) *verts[:, 0]

# image_size = img_cv2.shape[:2][::-1]

# print(verts.shape)
# verts = verts[:,0,:] # select 0 index
# # index_finger_tip = vertices[320]
# # thumb_tip = vertices[744]

# print('camera_matrix', camera_matrix)
# print('camera_translation', cam_t)
# print('pred_cam_t_full', pred_cam_t_full)
# print('cam_t', cam_t)
# print('img_size', image_size)

# def get_xyzs_from_depth(
#     depth: np.ndarray,
#     xys: np.ndarray,
#     intrinsic,
#     radius: int = 1,
#     depth_scale: float = 0.001,
# ):
#     fx = intrinsic[0, 0]
#     fy = intrinsic[1, 1]
#     cx = intrinsic[0, 2]
#     cy = intrinsic[1, 2]

#     if len(xys.shape) != 2:
#         xys = np.array([xys])

#     xs, ys = xys[:, 0], xys[:, 1]
#     if radius <= 1:
#         zs = depth[ys.astype(int), xs.astype(int)] * depth_scale
#     # else:
#     #     zs = circle_average(depth, xys, radius, np.median) * depth_scale

#     xs = (xs - cx) / fx * zs
#     ys = (ys - cy) / fy * zs

#     indices = np.where(zs != 0.0)

#     return np.array([xs[indices], ys[indices], zs[indices]]).T

# def project_to_2d(vertices, K):
#     vertices_2d = []
#     for vertex in vertices:
#         projected = K @ (vertex + pred_cam_t_full).T
#         projected = projected / projected[-1]
#         vertices_2d.append(projected[:2])
#     return np.array(vertices_2d)

# vertices_2d_1 = project_to_2d(verts, camera_matrix).reshape(-1, 2)
# print(vertices_2d_1)
# print('real: ', get_xyzs_from_depth(depth, vertices_2d_1, camera_matrix))
# print('pred: ', np.add(verts, pred_cam_t_full))