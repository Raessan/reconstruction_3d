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

    auto pcd = io::CreatePointCloudFromFile("../../data/data_pikachu/merged_pointcloud/merged_pc.ply");
    int depth = 5;

    std::cout << "Reconstructing..." << std::endl;

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

    // Visualize the mesh
    visualization::DrawGeometries({mesh}, "Filtered Poisson Mesh");

    return 0;
}