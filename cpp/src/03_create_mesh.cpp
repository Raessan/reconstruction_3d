#include <open3d/Open3D.h>
#include <iostream>
#include <filesystem>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "reconstruction/utils_pointcloud.hpp"
#include "reconstruction/utils_file.hpp"

namespace fs = std::filesystem;

using namespace open3d;
using namespace open3d::geometry;
using namespace open3d::pipelines::registration;

int main() {

    // Set the directory where your merged pc is stored and where the mesh will be saved
    std::string data_dir = "../../data/data_dummy";
    std::string merged_pc_dir = "merged_pointcloud";
    std::string mesh_dir = "mesh";

    // Depth for the reconstruction with Poisson
    int depth = 9;

    // Create directory for merged PC
    fs::create_directories(data_dir + "/" + mesh_dir);

    // Read the pcd
    auto pcd = io::CreatePointCloudFromFile(data_dir + "/" + merged_pc_dir + "/merged_pc.ply");
    

    std::cout << "Creating mesh..." << std::endl;

    // Perform Poisson Surface Reconstruction
    auto [mesh, densities] = geometry::TriangleMesh::CreateFromPointCloudPoisson(*pcd, static_cast<int>(depth));

    // Convert densities to a vector
    std::vector<double> densities_vec(densities.begin(), densities.end());

    // Compute density threshold (e.g., keep top 90% densest areas)
    std::nth_element(densities_vec.begin(), densities_vec.begin() + densities_vec.size() * 0.005, densities_vec.end());
    double density_threshold = densities_vec[densities_vec.size() * 0.005];

    // Identify vertices to remove
    std::vector<bool> vertices_to_remove(densities.size(), false);
    for (size_t i = 0; i < densities.size(); ++i) {
        if (densities[i] < density_threshold) {
            vertices_to_remove[i] = true;
        }
    }

    // Remove low-density vertices
    mesh->RemoveVerticesByMask(vertices_to_remove);

    // Save mesh
    open3d::io::WriteTriangleMesh(data_dir + "/" + merged_pc_dir + "/mesh.ply", *mesh);

    // Visualize the mesh
    visualization::DrawGeometries({mesh}, "Filtered Poisson Mesh");

    return 0;
}