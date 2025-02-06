import open3d as o3d
import copy
import numpy as np
import os

# Function to remove outliers
def outlier_removal(pc, nb_neighbors=1000, std_ratio=0.8):
    pc, ind = pc.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pc

# Function to voxelize the pointcloud
def voxelize(pcd, voxel_size=0.001):
    voxel_size = voxel_size  # Adjust voxel size according to your needs
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    return downsampled_pcd

# This function returns the biggest cluster
def biggest_cluster(pcd):
    # Apply Euclidean clustering
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=1000, print_progress=True))

    # Find the largest cluster
    max_label = labels.max()
    clusters = [pcd.select_by_index(np.where(labels == i)[0]) for i in range(max_label + 1)]
    largest_cluster = max(clusters, key=lambda x: len(x.points))
    return largest_cluster

# Function to estimate the normals
def estimate_normals(pointcloud, search_param_radius=0.05, fast=False):
    
    # Check if the input is a valid Open3D PointCloud
    assert isinstance(pointcloud, o3d.geometry.PointCloud), "Input must be an Open3D PointCloud"
    
    # Estimate normals
    pointcloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=search_param_radius, max_nn=30 if not fast else 10
        )
    )
    
    # Orient normals consistently
    pointcloud.orient_normals_consistent_tangent_plane(k=50)
    
    return pointcloud

# Function that preprocesses the pointcloud using the previous function
def preprocess_pcd(pcd, voxel_size = 0.001, with_normals = False):
    if with_normals:
        return estimate_normals(outlier_removal(voxelize(pcd, voxel_size)))
    return outlier_removal(voxelize(pcd, voxel_size)) #estimate_normals(outlier_removal(voxelize(pcd, voxel_size)))

# Function that applies a rotation and translation to a pointcloud
def transform_point_cloud(pointcloud, rotation, translation):
    
    ####### Apply rotation and translation to a point cloud and retain color #####
    new_pointcloud = copy.deepcopy(pointcloud)
    # Rotate and translate the 3D points
    # Convert point cloud to numpy array
    points = np.asarray(new_pointcloud.points)
    
    transformed_points = np.dot(points, rotation.T) + translation.T
    new_pointcloud.points = o3d.utility.Vector3dVector(transformed_points)
    return new_pointcloud

# Initialize the Open3D visualizer
def initialize_visualizer(width = 800, height = 600, title="Dynamic Point Cloud Visualization"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(title, width=width, height=height)
    return vis

# Update the point cloud in the visualizer
def update_visualizer(vis, point_cloud_list):
    vis.clear_geometries()
    for pc in point_cloud_list:
        vis.add_geometry(pc)
    vis.poll_events()
    vis.update_renderer()

# Function that performs ICP in two steps, first a coarse approximation and then fine
def run_icp(pcd1, pcd2, rotation, translation, coarse_threshold, fine_threshold):
    # Construct the initial transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = translation

    # Coarse registration
    reg_coarse = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, coarse_threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # Fine registration
    reg_fine = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, fine_threshold, reg_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # Information of the ICP
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        pcd1, pcd2, fine_threshold, reg_fine.transformation)

    # Return the refined transformation
    return transform_point_cloud(pcd1, reg_fine.transformation[:3, :3], reg_fine.transformation[:3, 3]), reg_fine.transformation, information_icp

# This function improves the initial guess of transformations individually
def improve_transform(pointclouds, coarse_threshold, fine_threshold, rotations = None, translations = None):

    N = len(pointclouds)

    # If the initial guesses are None, they are set to 3x3 Identity matrix and 3x1 vector, respectively
    if rotations is None:
        rotations = [np.eye(3)]*(N-1)
    if translations is None:
        translations = [np.zeros((3,1))]*(N-1)
    if len(rotations) != N - 1 or len(translations) != N - 1:
        raise ValueError("The number of rotations and translations must be N-1.")
        
    # Initialize reconstructed point clouds
    new_rotations = []
    new_translations = []

    # Apply the ICP to each pair of pointclouds
    for i in range(0, N-1):
        _, transformation, _ = run_icp(pointclouds[i],pointclouds[i+1],rotations[i],translations[i].squeeze(), coarse_threshold, fine_threshold)
        rot = transformation[:3,:3]
        trans = transformation[:3, 3]
        
        new_rotations.append(rot)
        new_translations.append(trans)
    
    return new_rotations, new_translations

# This function gets the transformation between two pointclouds, not necessarily subsequent. The indices should be provided with the last to arguments and idx_pc_2>idx_pc_1
def get_transformation(rotations, translations, idx_pc_1, idx_pc_2):

    assert(idx_pc_2>idx_pc_1)

    # Initialize cumulative transformation
    cumulative_rotation = np.eye(3)  # Identity rotation matrix
    cumulative_translation = np.zeros(3)  # Zero translation vector
    
    # Apply transformations sequentially
    for i in range(idx_pc_1, idx_pc_2):
        cumulative_rotation = np.dot(rotations[i], cumulative_rotation)
        cumulative_translation = np.dot(rotations[i], cumulative_translation) + translations[i].squeeze()
        
    return cumulative_rotation, cumulative_translation


def full_registration(pcds, rotations, translations, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            init_rot, init_trans = get_transformation(rotations, translations, source_id, target_id)
            #init_rot = np.eye(3)
            #init_trans = np.zeros(3)
            _, transformation_icp, information_icp = run_icp(pcds[source_id], pcds[target_id], init_rot, init_trans, max_correspondence_distance_coarse, max_correspondence_distance_fine)
            
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph