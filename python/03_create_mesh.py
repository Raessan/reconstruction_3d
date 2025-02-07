import open3d as o3d
import numpy as np
import os
from utils.utils_pointcloud import *

# Set the directory where your merged pc is stored and where the mesh will be saved
DATA_DIR = "../data/data_rayquaza"
MERGED_PC_DIR = "merged_pointcloud"
MESH_DIR = "mesh"

# Depth for the reconstruction with Poisson
DEPTH = 5

# Creation of directory for the mesh
os.makedirs(os.path.join(DATA_DIR, MESH_DIR), exist_ok=True)

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

# Save mesh
o3d.io.write_triangle_mesh(os.path.join(DATA_DIR, MESH_DIR, "mesh.ply"), mesh)
    
# Plot the mesh
o3d.visualization.draw_geometries([mesh])

