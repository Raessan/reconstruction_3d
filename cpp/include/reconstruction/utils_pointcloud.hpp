#include <open3d/Open3D.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <unordered_map>

using namespace open3d;
using namespace std;
using namespace Eigen;

// Function to remove outliers
std::shared_ptr<open3d::geometry::PointCloud> outlier_removal(const open3d::geometry::PointCloud& pc, int nb_neighbors = 1000, double std_ratio = 0.8) {
    auto [filtered_pcd, ind] = pc.RemoveStatisticalOutliers(nb_neighbors, std_ratio);
    return filtered_pcd;
}
// Function to voxelize the point cloud
std::shared_ptr<open3d::geometry::PointCloud> voxelize(const open3d::geometry::PointCloud& pcd, double voxel_size = 0.001) {
    return pcd.VoxelDownSample(voxel_size);
}

// Function to estimate normals
std::shared_ptr<open3d::geometry::PointCloud> estimate_normals(std::shared_ptr<open3d::geometry::PointCloud> pointcloud, double search_param_radius = 0.05, bool fast = false) {
    pointcloud->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(search_param_radius, fast ? 10 : 30));
    pointcloud->OrientNormalsConsistentTangentPlane(50);
    return pointcloud;
}

// Function to preprocess the point cloud
std::shared_ptr<open3d::geometry::PointCloud> preprocess_pcd(const geometry::PointCloud &pcd, double voxel_size = 0.001, bool with_normals = false) {
    auto downsampled = voxelize(pcd, voxel_size);
    auto outliers_removed = outlier_removal(*downsampled);
    if (with_normals) return estimate_normals(outliers_removed);
    else return outliers_removed;
}

// Function to transform a point cloud
std::shared_ptr<open3d::geometry::PointCloud> transform_pointcloud(const open3d::geometry::PointCloud& pointcloud, const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation) {
    auto transformed_pcd = std::make_shared<open3d::geometry::PointCloud>(pointcloud);
    std::vector<Vector3d> points = pointcloud.points_;
    for (auto &point : points){
        point = (rotation * point) + translation;
    }
    transformed_pcd->points_ = points;
    return transformed_pcd;
}

// Function to run ICP
std::tuple<std::shared_ptr<open3d::geometry::PointCloud>, Eigen::Matrix4d, Eigen::MatrixXd>
run_icp(const open3d::geometry::PointCloud& pcd1, const open3d::geometry::PointCloud& pcd2, 
       const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation, 
       double coarse_threshold, double fine_threshold) {
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    transformation.block<3, 3>(0, 0) = rotation;
    transformation.block<3, 1>(0, 3) = translation;

    auto reg_coarse = open3d::pipelines::registration::RegistrationICP(
        pcd1, pcd2, coarse_threshold, transformation,
        open3d::pipelines::registration::TransformationEstimationPointToPoint());

    // For fine registration, we use PointToPoint or PointToPlane according to whether the PCs have normals

    // Check if both point clouds have normals
    bool has_normals = pcd1.HasNormals() && pcd2.HasNormals();

    // Select fine registration method
    std::shared_ptr<open3d::pipelines::registration::TransformationEstimation> estimation_method;
    if (has_normals) {
        estimation_method = std::make_shared<open3d::pipelines::registration::TransformationEstimationPointToPlane>();
    } else {
        estimation_method = std::make_shared<open3d::pipelines::registration::TransformationEstimationPointToPoint>();
    }

    // Fine registration
    auto reg_fine = open3d::pipelines::registration::RegistrationICP(
        pcd1, pcd2, fine_threshold, reg_coarse.transformation_, *estimation_method);

    auto information_icp = open3d::pipelines::registration::GetInformationMatrixFromPointClouds(
        pcd1, pcd2, fine_threshold, reg_fine.transformation_);

    auto transformed_pcd = transform_pointcloud(pcd1, reg_fine.transformation_.block<3, 3>(0, 0), reg_fine.transformation_.block<3, 1>(0, 3));
    return {transformed_pcd, reg_fine.transformation_, information_icp};
}

// Function to improve transformations
std::tuple<std::vector<Eigen::Matrix3d>, std::vector<Eigen::Vector3d>>
improve_transform(const std::vector<open3d::geometry::PointCloud>& pointclouds, 
                 double coarse_threshold, double fine_threshold,
                 std::vector<Eigen::Matrix3d> rotations = {}, std::vector<Eigen::Vector3d> translations = {}) {
    int N = pointclouds.size();

    if (rotations.empty()) rotations.assign(N - 1, Eigen::Matrix3d::Identity());
    if (translations.empty()) translations.assign(N - 1, Eigen::Vector3d::Zero());

    if (rotations.size() != N - 1 || translations.size() != N - 1) {
        throw std::invalid_argument("The number of rotations and translations must be N-1.");
    }

    std::vector<Eigen::Matrix3d> new_rotations;
    std::vector<Eigen::Vector3d> new_translations;

    // std::shared_ptr<open3d::geometry::PointCloud> pc_aux;
    // Eigen::Matrix4d transformation;
    // Eigen::MatrixXd information;

    for (int i = 0; i < N - 1; ++i) {
        auto [pc_aux, transformation, information] = run_icp(pointclouds[i], pointclouds[i + 1], rotations[i], translations[i], coarse_threshold, fine_threshold);
        new_rotations.push_back(transformation.block<3, 3>(0, 0));
        new_translations.push_back(transformation.block<3, 1>(0, 3));
    }

    return {new_rotations, new_translations};
}

// Function to get transformations between two point clouds
tuple<Matrix3d, Vector3d> get_transformation(const vector<Matrix3d> &rotations,
                                            const vector<Vector3d> &translations,
                                            size_t idx1, size_t idx2) {
    assert(idx2 > idx1);

    Matrix3d cumulative_rotation = Matrix3d::Identity();
    Vector3d cumulative_translation = Vector3d::Zero();

    for (size_t i = idx1; i < idx2; ++i) {
        cumulative_rotation = rotations[i] * cumulative_rotation;
        cumulative_translation = rotations[i] * cumulative_translation + translations[i];
    }

    return {cumulative_rotation, cumulative_translation};
}

// Function for full registration
pipelines::registration::PoseGraph full_registration(const vector<geometry::PointCloud> &pcds,
                                                    const vector<Matrix3d> &rotations,
                                                    const vector<Vector3d> &translations,
                                                    double coarse_dist, double fine_dist) {
    pipelines::registration::PoseGraph pose_graph;
    Matrix4d odometry = Matrix4d::Identity();
    pose_graph.nodes_.emplace_back(pipelines::registration::PoseGraphNode(odometry));

    size_t N = pcds.size();
    for (size_t source_id = 0; source_id < N; ++source_id) {
        for (size_t target_id = source_id + 1; target_id < N; ++target_id) {
            auto [init_rot, init_trans] = get_transformation(rotations, translations, source_id, target_id);

            auto [_, transformation_icp, information_icp] = run_icp(
                pcds[source_id], pcds[target_id], init_rot, init_trans,
                coarse_dist, fine_dist);

            if (target_id == source_id + 1) {
                odometry = transformation_icp * odometry;
                pose_graph.nodes_.emplace_back(
                    pipelines::registration::PoseGraphNode(odometry.inverse()));
                pose_graph.edges_.emplace_back(
                    pipelines::registration::PoseGraphEdge(
                        source_id, target_id, transformation_icp, information_icp, false));
            } else {
                pose_graph.edges_.emplace_back(
                    pipelines::registration::PoseGraphEdge(
                        source_id, target_id, transformation_icp, information_icp, true));
            }
        }
    }
    return pose_graph;
}