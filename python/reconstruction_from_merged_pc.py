import open3d as o3d
import numpy as np
import os
from utils.utils_pointcloud import *

# Set the directory where your .ply files are stored
DATA_DIR = "../data/data_totodile"
MERGED_PC_DIR = "merged_pointcloud"

# Depth for the reconstruction with Poisson
DEPTH = 5

# Read the point clouds
merged_pcd = o3d.io.read_point_cloud(os.path.join(DATA_DIR, MERGED_PC_DIR, "merged_pc.ply"))

print('run Poisson surface reconstruction')
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(merged_pcd, depth=DEPTH)

# Convert densities to a numpy array for filtering
densities = np.asarray(densities)

# Compute a density threshold (e.g., based on a percentile or manually chosen value)
density_threshold = np.percentile(densities, 0.5)  # Keep the top 90% of dense areas

# Remove low-density vertices
vertices_to_remove = densities < density_threshold
mesh.remove_vertices_by_mask(vertices_to_remove)
    
# Plot the mesh
o3d.visualization.draw_geometries([mesh])