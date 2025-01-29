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
    const std::string DATA_DIR = "../data_rayquaza";
    const std::string PC_DIR = "pointclouds";
    const std::string MERGED_PC_DIR = "merged_pointcloud";
    const std::string ROT_DIR = "rotations";
    const std::string TRANS_DIR = "translations";
    const double VOXEL_SIZE = 0.001;
    const double COARSE_THRESHOLD = VOXEL_SIZE * 15;
    const double FINE_THRESHOLD = VOXEL_SIZE * 5;

    int N_MERGE = 0;

    // Load point cloud files
    std::vector<std::string> pointcloud_files;
    for (const auto& entry : fs::directory_iterator(DATA_DIR + "/" + PC_DIR)) {
        pointcloud_files.push_back(entry.path().string());
    }
    std::sort(pointcloud_files.begin(), pointcloud_files.end());


    if (N_MERGE <= 0) {
        N_MERGE = pointcloud_files.size();
    }

    // Load and preprocess point clouds
    std::vector<PointCloud> pointclouds;
    for (size_t i = 0; i < N_MERGE; ++i) {
        auto pcd = io::CreatePointCloudFromFile(pointcloud_files[i]);
        pointclouds.push_back(*preprocess_pcd(*pcd, VOXEL_SIZE));
    }


    std::vector<std::string> rotation_files;
    for (const auto& entry : fs::directory_iterator(DATA_DIR + "/" + ROT_DIR)) {
        rotation_files.push_back(entry.path().string());
    }
    std::sort(rotation_files.begin(), rotation_files.end());

    // Load transformations (rotations and translations)
    std::vector<Eigen::Matrix3d> rotations(N_MERGE-1);
    for (int i=0; i<N_MERGE-1; i++){
        load_rotation(rotation_files[i], rotations[i]);
    }

    std::vector<std::string> translation_files;
    for (const auto& entry : fs::directory_iterator(DATA_DIR + "/" + TRANS_DIR)) {
        translation_files.push_back(entry.path().string());
    }
    std::sort(translation_files.begin(), translation_files.end());

    // Load transformations (rotations and translations)
    std::vector<Eigen::Vector3d> translations(N_MERGE-1);
    for (int i=0; i<N_MERGE-1; i++){
        load_translation(translation_files[i], translations[i]);
    }

    std::cout << "Calculating improved transform" << std::endl;
    auto [new_rot, new_trans] = improve_transform(pointclouds, COARSE_THRESHOLD, FINE_THRESHOLD, rotations, translations);

    // Pose graph optimization
    std::cout << "Creating PoseGraph..." << std::endl;
    auto pose_graph = full_registration(pointclouds, new_rot, new_trans, COARSE_THRESHOLD, FINE_THRESHOLD);

    std::cout << "Optimizing PoseGraph..." << std::endl;
    GlobalOptimizationOption option(FINE_THRESHOLD, 0.25, 0);
    GlobalOptimization(pose_graph, GlobalOptimizationLevenbergMarquardt(),
                        GlobalOptimizationConvergenceCriteria(), option);

    // Transform points based on pose graph
    std::cout << "Transforming PoseGraph..." << std::endl;
    for (size_t i = 0; i < pointclouds.size(); ++i) {
        pointclouds[i].Transform(pose_graph.nodes_[i].pose_);
    }

    // Outlier removal
    std::cout << "Remove outliers..." << std::endl;
    std::vector<PointCloud> processed_pcds;
    for (const auto& pcd : pointclouds) {
        auto downsampled = voxelize(pcd, VOXEL_SIZE);
        processed_pcds.push_back(*outlier_removal(*downsampled, 20, 2.0));
    }

    // Merge point clouds
    std::cout << "Merging pointclouds..." << std::endl;
    auto merged_pcd = std::make_shared<PointCloud>();
    for (const auto& pcd : processed_pcds) {
        *merged_pcd += pcd;
    }

    if (open3d::io::WritePointCloud(DATA_DIR + "/" + MERGED_PC_DIR + "/merged_pc.ply", *merged_pcd)){
        std::cout << "Merged point cloud saved successfully!" << std::endl;
    }
    else{
        std::cerr << "Failed to save point cloud!" << std::endl;
    }
    visualization::DrawGeometries({merged_pcd});

    // Poisson surface reconstruction
    std::cout << "Running Poisson surface reconstruction..." << std::endl;
    auto [mesh, densities] = TriangleMesh::CreateFromPointCloudPoisson(*merged_pcd, 8);

    // // Convert densities to filter vertices
    // double density_threshold = densities->GetMinBound() + 0.5 * (densities->GetMaxBound() - densities->GetMinBound());
    // mesh->RemoveVerticesByMask(densities->ComputeDensePointsMask(density_threshold));

    // Visualize the mesh
    std::cout << "Visualizing..." << std::endl;
    visualization::DrawGeometries({mesh});

    return 0;
}