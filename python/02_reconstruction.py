import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from utils.utils_pointcloud import *

# Set the directory where your .ply files are stored
DATA_DIR = "../data/data_rayquaza"
PC_DIR = "pointclouds"
ROT_DIR = "rotations"
TRANS_DIR = "translations"
MERGED_PC_DIR = "merged_pointcloud"

# Variables for voxelization
VOXEL_SIZE = 0.001
COARSE_THRESHOLD = VOXEL_SIZE*15
FINE_THRESHOLD = VOXEL_SIZE*5

# Number of pointclouds to merge. If set to 0, it considers all the pointclouds. Otherwise, it considers the first N_MERGE pointclouds
N_MERGE = 0

# Creation of directory for the merged PC
os.makedirs(os.path.join(DATA_DIR, MERGED_PC_DIR), exist_ok=True)

# Load all .ply files
pointcloud_files = [f for f in sorted(os.listdir(os.path.join(DATA_DIR, PC_DIR)))]
print(pointcloud_files)
if N_MERGE <= 0:
    N_MERGE = len(pointcloud_files)

# Read the point clouds
pointclouds = [preprocess_pcd(o3d.io.read_point_cloud(os.path.join(DATA_DIR, PC_DIR, f)), voxel_size=VOXEL_SIZE, with_normals=True) for f in pointcloud_files[:N_MERGE]]

rotation_files = [f for f in sorted(os.listdir(os.path.join(DATA_DIR, ROT_DIR)))]
print(rotation_files)
#rotations = [np.load(os.path.join(DATA_DIR, ROT_DIR, f)) for f in rotation_files[:(N_MERGE-1)]]
rotations = [np.loadtxt(os.path.join(DATA_DIR, ROT_DIR, f)) for f in rotation_files[:(N_MERGE-1)]]

trans_files = [f for f in sorted(os.listdir(os.path.join(DATA_DIR, TRANS_DIR)))]
print(trans_files)
#translations = [np.load(os.path.join(DATA_DIR, TRANS_DIR, f)) for f in trans_files[:(N_MERGE-1)]]
translations = [np.loadtxt(os.path.join(DATA_DIR, TRANS_DIR, f)) for f in trans_files[:(N_MERGE-1)]]

# Initial rotation and translation improvement
new_rot, new_trans = improve_transform(pointclouds, COARSE_THRESHOLD, FINE_THRESHOLD, rotations, translations)

# Pose graph optimization
print("Creating PoseGraph ...")
pose_graph = full_registration(pointclouds, new_rot, new_trans, COARSE_THRESHOLD, FINE_THRESHOLD)
        
print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=FINE_THRESHOLD,
    edge_prune_threshold=0.25,
    reference_node=0)

o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

pointcloud_graph = copy.deepcopy(pointclouds)

print("Transform points and display")
for point_id in range(len(pointclouds)):
    pointcloud_graph[point_id].transform(pose_graph.nodes[point_id].pose)


# We perform outier removal on the pointclouds
pointcloud_graph_processed = []
for pc in pointcloud_graph:
    pointcloud_graph_processed.append(outlier_removal(voxelize(pc, VOXEL_SIZE), nb_neighbors=20, std_ratio=2.0))

# We create the mesh with percentile for dense areas to avoid strange surfaces
merged_pcd = o3d.geometry.PointCloud()
for pcd in pointcloud_graph_processed:
    merged_pcd += pcd

# Save merged pcd
print("Saving merged PC")
o3d.io.write_point_cloud(os.path.join(DATA_DIR, MERGED_PC_DIR, "merged_pc.ply"), merged_pcd)

print("Plotting merged pointcloud")
o3d.visualization.draw_geometries([merged_pcd])