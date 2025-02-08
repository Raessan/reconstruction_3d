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

    // Variables for the path and directory names
    const std::string data_dir = "../../data/data_dummy";
    const std::string pc_dir = "pointclouds";
    const std::string merged_pc_dir = "merged_pointcloud";
    const std::string rot_dir = "rotations";
    const std::string trans_dir = "translations";

    // Variables for the pointcloud processing
    const double voxel_size = 0.001;
    const double coarse_threshold = voxel_size * 15;
    const double fine_threshold = voxel_size * 5;

    // If n_merge is 0, all the pointclouds from the folder are used, otherwise, it will use the first n_merge pointclouds
    int n_merge = 0;

    // Create directory for merged PC
    fs::create_directories(data_dir + "/" + merged_pc_dir);

    // Load point cloud files
    std::vector<std::string> pointcloud_files;
    for (const auto& entry : fs::directory_iterator(data_dir + "/" + pc_dir)) {
        pointcloud_files.push_back(entry.path().string());
    }
    std::sort(pointcloud_files.begin(), pointcloud_files.end());


    if (n_merge <= 0) {
        n_merge = pointcloud_files.size();
    }

    // Load and preprocess point clouds
    std::vector<PointCloud> pointclouds;
    for (size_t i = 0; i < n_merge; ++i) {
        auto pcd = io::CreatePointCloudFromFile(pointcloud_files[i]);
        pointclouds.push_back(*preprocess_pcd(*pcd, voxel_size, true));
    }


    std::vector<std::string> rotation_files;
    for (const auto& entry : fs::directory_iterator(data_dir + "/" + rot_dir)) {
        rotation_files.push_back(entry.path().string());
    }
    std::sort(rotation_files.begin(), rotation_files.end());

    // Load transformations (rotations and translations)
    std::vector<Eigen::Matrix3d> rotations(n_merge-1);
    for (int i=0; i<n_merge-1; i++){
        load_rotation(rotation_files[i], rotations[i]);
    }

    std::vector<std::string> translation_files;
    for (const auto& entry : fs::directory_iterator(data_dir + "/" + trans_dir)) {
        translation_files.push_back(entry.path().string());
    }
    std::sort(translation_files.begin(), translation_files.end());

    // Load transformations (rotations and translations)
    std::vector<Eigen::Vector3d> translations(n_merge-1);
    for (int i=0; i<n_merge-1; i++){
        load_translation(translation_files[i], translations[i]);
    }

    std::cout << "Calculating improved transform" << std::endl;
    auto [new_rot, new_trans] = improve_transform(pointclouds, coarse_threshold, fine_threshold, rotations, translations);

    // Pose graph optimization
    std::cout << "Creating PoseGraph..." << std::endl;
    auto pose_graph = full_registration(pointclouds, new_rot, new_trans, coarse_threshold, fine_threshold);

    std::cout << "Optimizing PoseGraph..." << std::endl;
    GlobalOptimizationOption option(fine_threshold, 0.25, 1.0, 0);
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
        auto downsampled = voxelize(pcd, voxel_size);
        processed_pcds.push_back(*outlier_removal(*downsampled, 20, 2.0));
    }

    // Merge point clouds
    std::cout << "Merging pointclouds..." << std::endl;
    auto merged_pcd = std::make_shared<PointCloud>();
    for (const auto& pcd : processed_pcds) {
        *merged_pcd += pcd;
    }


    if (open3d::io::WritePointCloud(data_dir + "/" + merged_pc_dir + "/merged_pc.ply", *merged_pcd)){
        std::cout << "Merged point cloud saved successfully!" << std::endl;
    }
    else{
        std::cerr << "Failed to save point cloud!" << std::endl;
    }
    
    visualization::DrawGeometries({merged_pcd});

    return 0;
}