import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import random


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


def cam2world(cam_coord, R, t):
    world_coord = np.dot(
        np.linalg.inv(R), (cam_coord - t.reshape(1, 3)).transpose(1, 0)
    ).transpose(1, 0)
    return world_coord


def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)
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
        X = np.dot(B, A_inv)  # X = B * A-1 ~~ XA = B
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


def exo2ego(
    exo_rgb_path,
    exo_depth_path,
    exo_cam_int_path,
    exo_cam_ext_path,
    exo_hand_path,
    ego_cam_int_path,
    ego_cam_ext_path,
    ego_hand_path,
):

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
    X = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    ego_cam_int = np.loadtxt(ego_cam_int_path)
    ego_fx, ego_fy, ego_cx, ego_cy, ego_w, ego_h = ego_cam_int
    ego_K = np.array([[ego_fx, 0, ego_cx], [0, ego_fy, ego_cy], [0, 0, 1]])
    ego_rgb_pred = points2image(translated_points, colors, X, ego_K, ego_w, ego_h)

    return ego_rgb_pred


# root_path = '/local/home/andrye/dev/H2O'
# exo_cam_id = 'cam2'
# ego_cam_id  ='cam4'
# exo_rgb_paths = sorted(glob(os.path.join(root_path, 'subject1', '*/*', exo_cam_id, 'rgb/*.png')))
# # random.shuffle(exo_rgb_paths)
# for exo_rgb_path in tqdm(exo_rgb_paths):
#     # exos

exo_rgb_path = "/local/home/andrye/dev/H2O/subject1/h1/2/cam2/rgb/000043.png"
exo_cam_id = "cam2"
ego_cam_id = "cam4"

exo_rgb_path = exo_rgb_path.replace(exo_cam_id, exo_cam_id)
exo_depth_path = exo_rgb_path.replace("rgb", "depth")
exo_rgb_id = os.path.join("rgb", exo_rgb_path.split("/")[-1])
exo_cam_int_path = exo_rgb_path.replace(exo_rgb_id, "cam_intrinsics.txt")
exo_cam_ext_path = exo_rgb_path.replace("rgb", "cam_pose").replace("png", "txt")
exo_hand_path = exo_rgb_path.replace("rgb", "hand_pose").replace("png", "txt")
# egos
ego_cam_int_path = exo_cam_int_path.replace(exo_cam_id, ego_cam_id)
ego_cam_ext_path = exo_cam_ext_path.replace(exo_cam_id, ego_cam_id)
ego_hand_path = exo_hand_path.replace(exo_cam_id, ego_cam_id)
ego_rgb_pred = exo2ego(
    exo_rgb_path,
    exo_depth_path,
    exo_cam_int_path,
    exo_cam_ext_path,
    exo_hand_path,
    ego_cam_int_path,
    ego_cam_ext_path,
    ego_hand_path,
)
# vis
exo_rgb_gt = cv2.imread(exo_rgb_path)
ego_rgb_gt = cv2.imread(exo_rgb_path.replace(exo_cam_id, ego_cam_id))
cv2.imwrite("vis_sparse_ego.png", ego_rgb_pred)
