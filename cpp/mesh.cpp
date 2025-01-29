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

    auto pcd = io::CreatePointCloudFromFile("../data_rayquaza/merged_pointcloud/merged_pc.ply");
    //auto downsampled_pcd = biggest_cluster(*pcd);

    //std::cout << "Running Poisson surface reconstruction..." << std::endl;
    //auto [mesh, densities] = TriangleMesh::CreateFromPointCloudPoisson(*pcd, 9);

    // Create a KDTree for the point cloud
    auto pcd_tree = std::make_shared<open3d::geometry::KDTreeFlann>(*pcd);

    // Set ball radii (adjust as needed for your data)
    std::vector<double> radii = {0.005, 0.01, 0.02}; // Ball radii for pivoting

    // Perform ball-pivoting
    auto mesh = open3d::geometry::TriangleMesh::CreateFromPointCloudBallPivoting(
        *pcd, radii);

    // Visualize the mesh
    std::cout << "Visualizing..." << std::endl;
    visualization::DrawGeometries({mesh});

    return 0;
}