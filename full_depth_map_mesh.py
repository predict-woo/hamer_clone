from coords import *
import numpy as np
import trimesh
import open3d as o3d
import teaserpp_python

NOISE_BOUND = 0.05
N_OUTLIERS = 1700
OUTLIER_TRANSLATION_LB = 5
OUTLIER_TRANSLATION_UB = 10

# root_path = '/cluster/project/cvg/data/H2O'
# subject = 'subject4'
# exo_cam_id = 'cam2'
# ego_cam_id = 'cam4'

# print test exo_rgb_paths
# exo_rgb_path, exo_depth_path, exo_hand_path, exo_cam_ext_path, exo_cam_int_path = load_rgb_depth_cam_ext_int(root_path, subject, exo_cam_id, 10)
# ego_rgb_path, ego_depth_path, ego_hand_path, ego_cam_ext_path, ego_cam_int_path = load_rgb_depth_cam_ext_int(root_path, subject, ego_cam_id, 10)


exo_rgb_path = '/cluster/project/cvg/data/H2O/subject4/h2/3/cam2/rgb/000022.png'
exo_depth_path = '/cluster/project/cvg/data/H2O/subject4/h2/3/cam2/depth/000022.png'
exo_cam_int_path = '/cluster/project/cvg/data/H2O/subject4/h2/3/cam2/cam_intrinsics.txt'
exo_cam_ext_path = '/cluster/project/cvg/data/H2O/subject4/h2/3/cam2/cam_pose/000022.txt'
exo_mask_path = 'demo_out/exo_all_mask.png'

ego_rgb_path = '/cluster/project/cvg/data/H2O/subject4/h2/3/cam4/rgb/000022.png'
ego_depth_path = '/cluster/project/cvg/data/H2O/subject4/h2/3/cam4/depth/000022.png'
ego_cam_int_path = '/cluster/project/cvg/data/H2O/subject4/h2/3/cam4/cam_intrinsics.txt'
ego_cam_ext_path = '/cluster/project/cvg/data/H2O/subject4/h2/3/cam4/cam_pose/000022.txt'
ego_mask_path = 'demo_out/ego_all_mask.png'


def depth2obj(depth_path, rgb_path, mask_path, cam_int_path, cam_ext_path, prefix=None):
    cam_int = np.loadtxt(cam_int_path)
    fx, fy, cx, cy = cam_int[:4]
    cam_ext = np.loadtxt(cam_ext_path).reshape(4, 4)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    points = depth2points(depth, fx, fy, cx, cy).reshape(-1, 3)
    
    mask = mask.astype(bool).reshape(-1)
    points = points[~mask]

    
    colors = cv2.imread(rgb_path).reshape(-1, 3)
    colors = colors[~mask]

    R = cam_ext[:3, :3]
    t = cam_ext[:3, 3:]

    points /= 1000
    
    return points, colors




exo_points, exo_colors = depth2obj(exo_depth_path, exo_rgb_path, exo_mask_path, exo_cam_int_path, exo_cam_ext_path, 'exo')

point_count = exo_points.shape[0]
print(exo_points.shape, exo_colors.shape)

# mesh_exo.obj
exo_mesh = o3d.io.read_triangle_mesh('mesh_exo.obj')
exo_mesh.compute_vertex_normals()
pcd = exo_mesh.sample_points_uniformly(number_of_points=point_count)

vertices = np.asarray(pcd.points)

solver_params = teaserpp_python.RobustRegistrationSolver.Params()
solver_params.cbar2 = 1
solver_params.noise_bound = NOISE_BOUND
solver_params.estimate_scaling = False
solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
solver_params.rotation_gnc_factor = 1.4
solver_params.rotation_max_iterations = 100
solver_params.rotation_cost_threshold = 1e-12

solver = teaserpp_python.RobustRegistrationSolver(solver_params)
solver.solve(exo_points, vertices)

solution = solver.getSolution()

print(solution.rotation.shape)
print(solution.translation.shape)

# transform exo_points
exo_points_transformed = solution.rotation @ exo_points.T + solution.translation
exo_points_transformed = exo_points_transformed.T

# save as ply
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(exo_points_transformed)
o3d.io.write_point_cloud('exo_points_transformed.ply', pcd)

# print(vertices.shape, faces.shape)



# ego_world_points, ego_colors = depth2obj(ego_depth_path, ego_rgb_path, ego_mask_path, ego_cam_int_path, ego_cam_ext_path, 'ego')

# # concat 
# world_points = np.concatenate([exo_world_points, ego_world_points], axis=0)
# colors = np.concatenate([exo_colors, ego_colors], axis=0)

# world_pointcloud = trimesh.PointCloud(vertices=world_points, colors=colors)
# world_pointcloud.export('full_points.ply')

# # save as numpy
# np.save('exo_world_points.npy', exo_world_points)
# np.save('ego_world_points.npy', ego_world_points)
