from exo2ego import *
import numpy as np
import trimesh
root_path = '/local/home/andrye/dev/H2O'
subject = 'subject1'
exo_cam_id = 'cam2'
ego_cam_id  ='cam4'

# print test exo_rgb_paths
exo_rgb_path, exo_depth_path, exo_hand_path, exo_cam_ext_path, exo_cam_int_path = load_rgb_depth_cam_ext_int(root_path, subject, exo_cam_id, 10)
ego_rgb_path, ego_depth_path, ego_hand_path, ego_cam_ext_path, ego_cam_int_path = load_rgb_depth_cam_ext_int(root_path, subject, ego_cam_id, 10)

def depth2obj(depth_path, rgb_path, cam_int_path, cam_ext_path, prefix=None):
    cam_int = np.loadtxt(cam_int_path)
    fx, fy, cx, cy = cam_int[:4]
    cam_ext = np.loadtxt(cam_ext_path).reshape(4, 4)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    points = depth2points(depth, fx, fy, cx, cy).reshape(-1, 3)
    colors = cv2.imread(rgb_path).reshape(-1, 3)

    R = cam_ext[:3, :3]
    t = cam_ext[:3, 3:]

    points /= 1000

    # save as obj
    pointcloud = trimesh.PointCloud(vertices=points, colors=colors)
    pointcloud.export(f'{prefix}_points.ply')

    world_points = world2cam(points, R, t)

    # save as obj
    world_pointcloud = trimesh.PointCloud(vertices=world_points, colors=colors)
    world_pointcloud.export(f'{prefix}_points_world.ply')
    
    return world_points, colors


exo_world_points, exo_colors = depth2obj(exo_depth_path, exo_rgb_path, exo_cam_int_path, exo_cam_ext_path, 'exo')
ego_world_points, ego_colors = depth2obj(ego_depth_path, ego_rgb_path, ego_cam_int_path, ego_cam_ext_path, 'ego')


# save each pointcloud as ply
trimesh.PointCloud(vertices=exo_world_points, colors=exo_colors).export('exo_world_points.ply')
trimesh.PointCloud(vertices=ego_world_points, colors=ego_colors).export('ego_world_points.ply')


# # concat 
# world_points = np.concatenate([exo_world_points, ego_world_points], axis=0)
# colors = np.concatenate([exo_colors, ego_colors], axis=0)

# world_pointcloud = trimesh.PointCloud(vertices=world_points, colors=colors)
# world_pointcloud.export('full_points.ply')

# # save as numpy
# np.save('exo_world_points.npy', exo_world_points)
# np.save('ego_world_points.npy', ego_world_points)
