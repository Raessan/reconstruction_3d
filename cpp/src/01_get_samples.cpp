#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <memory>
#include <sstream>
#include <deque>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#include <open3d/Open3D.h>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "nn_handler_lib/nn_handler.hpp"

#include "reconstruction/utils_file.hpp"
#include "reconstruction/utils_sift.hpp"
#include "reconstruction/utils_pointcloud.hpp"


namespace fs = std::filesystem;

// Directories
/** \brief Directory for output  */
const std::string output_dir = "../../data/data_totodile";
/** \brief Directory for pointcloud inside output  */
const std::string pc_dir = "pointclouds";
/** \brief Directory for rotations inside output  */
const std::string rotations_dir = "rotations";
/** \brief Directory for translations inside output  */
const std::string translations_dir = "translations";

/** \brief Number of points to use for the inference */
int n_points = 5;

/** \brief Minimum depth to consider  */
constexpr double min_depth = 0.1; 
/** \brief Maximum depth to consider  */
constexpr double max_depth = 1.0; 
/** \brief Width of the original image  */
constexpr int w_realsense = 640; 
/** \brief Height of the original image  */
constexpr int h_realsense = 480; 
/** \brief Width NN image  */
constexpr int w_nn = 1024; 
/** \brief Height of NN image  */
constexpr int h_nn = 1024; 
/** \brief Width plot Open3D  */
constexpr int w_plot = 800; 
/** \brief Height plot Open3D  */
constexpr int h_plot = 600; 
/** \brief Voxel size  */
constexpr double voxel_size = 0.001; 
/** \brief Coarse threshold for ICP  */
constexpr double coarse_threshold = voxel_size*15; 
/** \brief Fine threshold for ICP  */
constexpr double fine_threshold = voxel_size*5; 


/** \brief Path of the encoder model*/
const std::string path_encoder = "/ssd/Datasets_and_code/reconstruction_project/models/sam2.1_hiera_small_encoder.onnx";
/** \brief Path of the decoder model */
const std::string path_decoder = "/ssd/Datasets_and_code/reconstruction_project/models/sam2.1_hiera_small_decoder.onnx";
/** \brief Path to save the TensorRT engine for inference*/
const std::string path_engine_save = "/ssd/Datasets_and_code/reconstruction_project/models/";

// DLA core to use. If -1, it does not use it
int dla_core = -1;
// GPU index (ORIN only has the 0 index)
int device_index=0;
// Batch size. If the model has fixed batch, this has to be 1. If the model has dynamic batch, this can be >1
int batch_size=1;

// Other variables needed for inference
/** \brief Pointer of encoder model */
std::unique_ptr<NNHandler> nn_handler_encoder;

/** \brief Pointer of decoder model */
std::unique_ptr<NNHandler> nn_handler_decoder;

/** \brief Vector of positive points */
std::vector<std::vector<int>> positive_points;
/** \brief Vector of negative points */
std::vector<std::vector<int>> negative_points;
/** \brief Merged_points and labels */
std::vector<std::vector<float>> merged_points, labels;

/** \brief Variables needed */
std::vector<cv::Mat> images, matched_images, images_segmented, depths;
std::vector<Eigen::Matrix3d> rotations;
std::vector<Eigen::Vector3d> translations;
std::vector<std::pair<std::vector<cv::KeyPoint>, cv::Mat>> features;

Eigen::Matrix3d R_eigen;
Eigen::Vector3d t_eigen;

/** \brief Variables for pointcloud */
std::vector<Eigen::Matrix3d> plot_rotations;
std::vector<Eigen::Vector3d> plot_translations;
std::vector<std::shared_ptr<open3d::geometry::PointCloud>> plot_pc;

void CreateDirectories() {
    fs::create_directories(output_dir + "/" + pc_dir);
    fs::create_directories(output_dir + "/" + rotations_dir);
    fs::create_directories(output_dir + "/" + translations_dir);
}

void manage_clicks(int event, int x, int y, int flags, void* param) {
    if (positive_points.size() + negative_points.size() < n_points){
        if (event == cv::EVENT_LBUTTONDOWN) {
            positive_points.push_back({x, y});
        } else if (event == cv::EVENT_RBUTTONDOWN) {
            negative_points.push_back({x, y});
        }
    }
}

void initialize_points(){
    positive_points.clear();
    negative_points.clear();
    positive_points.push_back({w_realsense/2, h_realsense/2});
    for (int i=0; i<n_points; i++){
        merged_points[0][2*i] = w_nn/2.0;
        merged_points[0][2*i+1] = h_nn/2.0;
        labels[0][i] = -1.0;
    }
    labels[0][0] = 1.0;
}

int main() {
    // Create directories
    CreateDirectories();
    // Initialize inferencers
    nn_handler_encoder = std::make_unique<NNHandler>(path_encoder, path_engine_save, Precision::FP16, dla_core, device_index, batch_size);
    nn_handler_decoder = std::make_unique<NNHandler>(path_decoder, path_engine_save, Precision::FP16, dla_core, device_index, batch_size);
    
    // Initialize RealSense pipeline
    rs2::pipeline pipeline;
    rs2::config config;
    config.enable_stream(RS2_STREAM_DEPTH, w_realsense, h_realsense, RS2_FORMAT_Z16, 30);
    config.enable_stream(RS2_STREAM_COLOR, w_realsense, h_realsense, RS2_FORMAT_BGR8, 30);
    
    // Start pipeline
    rs2::pipeline_profile profile = pipeline.start(config);

    // Get depth sensor scale
    rs2::device dev = profile.get_device();
    rs2::depth_sensor depth_sensor = dev.first<rs2::depth_sensor>();
    float depth_scale = depth_sensor.get_depth_scale();

    // Create pointcloud object
    rs2::pointcloud pc;
    rs2::align align(RS2_STREAM_COLOR);

    std::cout << "Press SPACE to capture an image and point cloud.\n";
    std::cout << "Press 'r' to remove previous sample.\n";
    std::cout << "Press 'd' to delete auxiliar points.\n";
    std::cout << "Press 'q' to quit the app.\n";

    cv::Mat complete_image(h_realsense * 2, w_realsense * 2, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat masked_image, matched_image;

    int counter = 0;
    
    // positive_points.push_back({w_realsense/2, h_realsense/2});
    merged_points.resize(1);
    merged_points[0].resize(n_points*2);
    labels.resize(1);
    labels[0].resize(n_points);

    initialize_points();

    // Open3D Visualizer for the point cloud
    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("PointCloud", w_plot, h_plot);

    // Merged PC for plot
    open3d::geometry::PointCloud merged_pc;

    while (true) {

        // Update visualizer
        vis.PollEvents();
        vis.UpdateRender();

        auto frames = pipeline.wait_for_frames();
        auto aligned_frames = align.process(frames);
        auto color_frame = aligned_frames.get_color_frame();
        auto depth_frame = aligned_frames.get_depth_frame();

        if (!depth_frame || !color_frame) continue;

        // Retrieve the stream profile for the color frame
        rs2::video_stream_profile color_stream = color_frame.get_profile().as<rs2::video_stream_profile>();

        // Get color camera intrinsics
        rs2_intrinsics color_intrinsics = color_stream.get_intrinsics();

        // Construct the camera matrix
        cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 
            color_intrinsics.fx, 0, color_intrinsics.ppx,
            0, color_intrinsics.fy, color_intrinsics.ppy,
            0, 0, 1);

        // Print the camera matrix
        //std::cout << "Camera Matrix:\n" << camera_matrix << std::endl;

        cv::Mat color_image_bgr(cv::Size(w_realsense, h_realsense), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat color_image_bgr_annotated = color_image_bgr.clone();

        for (const auto& pt : positive_points) {
            cv::circle(color_image_bgr_annotated, cv::Point(pt[0], pt[1]), 5, cv::Scalar(255, 0, 0), -1);
        }
        for (const auto& pt : negative_points) {
            cv::circle(color_image_bgr_annotated, cv::Point(pt[0], pt[1]), 5, cv::Scalar(0, 0, 255), -1);
        }

        color_image_bgr_annotated.copyTo(complete_image(cv::Rect(0, 0, w_realsense, h_realsense)));

        if (!masked_image.empty()){
            masked_image.copyTo(complete_image(cv::Rect(w_realsense, 0, w_realsense, h_realsense)));
        }
        else {
            // Set the rectangle region to black (0, 0, 0 for black in BGR)
            complete_image(cv::Rect(w_realsense, 0, w_realsense, h_realsense)).setTo(cv::Scalar(0, 0, 0));
        }

        if (!matched_image.empty()){
            matched_image.copyTo(complete_image(cv::Rect(0, h_realsense, 2*w_realsense, h_realsense)));
        }
        else {
            // Set the rectangle region to black (0, 0, 0 for black in BGR)
            complete_image(cv::Rect(0, h_realsense, 2 * w_realsense, h_realsense)).setTo(cv::Scalar(0, 0, 0));
        }

        cv::imshow("Complete Image", complete_image);
        cv::setMouseCallback("Complete Image", manage_clicks);

        // Convert depth frame to OpenCV matrix
        cv::Mat depth_mat(cv::Size(depth_frame.as<rs2::video_frame>().get_width(), 
                                depth_frame.as<rs2::video_frame>().get_height()), 
                                CV_16UC1, 
                                (void*)depth_frame.get_data(), 
                                cv::Mat::AUTO_STEP);

        // Convert depth to meters by multiplying with the depth scale
        cv::Mat depth;
        depth_mat.convertTo(depth, CV_32F, depth_scale);

        // Convert image from BGR to RGB
        cv::Mat color_image;
        cv::cvtColor(color_image_bgr, color_image, cv::COLOR_BGR2RGB);

        auto key = cv::waitKey(1);
        if ((char)key == 'q'){
            for (int i=0; i<rotations.size(); i++){
                std::ostringstream ss;
                ss << output_dir << "/" << rotations_dir << "/rotation_" << std::setw(2) << std::setfill('0') << i << ".txt";
                save_eigen_txt(rotations[i], ss.str());
            }
            for (int i=0; i<translations.size(); i++){
                std::ostringstream ss;
                ss << output_dir << "/" << translations_dir << "/translation_" << std::setw(2) << std::setfill('0') << i << ".txt";
                save_eigen_txt(translations[i], ss.str());
            }
            break;
        } 
        else if ((char)key == 'r'){
            initialize_points();

            if (counter>0){
                images.pop_back();
                depths.pop_back();
                images_segmented.pop_back();
                features.pop_back();
                std::ostringstream ss;
                ss << output_dir << "/" << pc_dir << "/pointclouds_" << std::setw(2) << std::setfill('0') << counter-1 << ".ply";
                if (std::remove(ss.str().c_str()) != 0) {
                    std::cerr << "Error deleting pointcloud file" << std::endl;
                }
                plot_rotations.pop_back();
                plot_translations.pop_back();
                plot_pc.pop_back();

                merged_pc.Clear();
                for (const auto &pc : plot_pc) merged_pc += *pc;

                vis.ClearGeometries();
                vis.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(merged_pc));
            }

            if (counter>1){
                matched_images.pop_back();
                rotations.pop_back();
                translations.pop_back();
            }

            if (counter>0){
                counter--;
            }

            if (counter>0) masked_image = images_segmented.back().clone();
            else masked_image.release();

            if (counter>1) matched_image = matched_images.back().clone();
            else matched_image.release();

        }
        else if ((char)key == 'd'){
            initialize_points();
        }
        else if ((char)key == 32){ // Press key

            /** \brief Merged_points */
            for (int i=0; i<positive_points.size(); i++){
                merged_points[0][2*i] = positive_points[i][0]*w_nn/w_realsense;
                merged_points[0][2*i+1] = positive_points[i][1]*h_nn/h_realsense;
                labels[0][i] = 1.0;
            }
            int n_positive_points = positive_points.size();
            for (int i=0; i<negative_points.size(); i++){
                merged_points[0][2*n_positive_points+2*i] = negative_points[i][0]*w_nn/w_realsense;
                merged_points[0][2*n_positive_points+2*i+1] = negative_points[i][1]*h_nn/h_realsense;
                labels[0][n_positive_points+i] = 0.0;
            }


            // Adapt data from image, so we have NCHW format, normalized and resized for the NN
            cv::Mat blob_im = cv::dnn::blobFromImage(color_image_bgr, 1.0/255.0, cv::Size(w_nn, h_nn), cv::Scalar(0.0, 0.0, 0.0), false, false, CV_32F);
            
            // Input and output of first NN
            float * input_encoder = blob_im.ptr<float>();
            std::vector<std::vector<std::vector<float>>> output_encoder;

            nn_handler_encoder->run_inference(input_encoder, output_encoder, false);

            std::swap(output_encoder[1], output_encoder[2]);

            output_encoder.push_back(merged_points);
            output_encoder.push_back(labels);

            std::vector<std::vector<std::vector<float>>> output_decoder;
            nn_handler_decoder->run_inference(output_encoder, output_decoder);

            cv::Mat mask = cv::Mat(output_decoder[1][0]).reshape(1, 256);
            cv::resize(mask, mask, cv::Size(w_realsense, h_realsense), 0, 0, cv::INTER_NEAREST);
            mask.convertTo(mask, CV_8U);


            initialize_points();
            
            // Process mask
            // mask = dilate_mask(mask, 5);
            mask = erode_mask(mask, 3);

            // Apply the mask to the image using bitwise AND
            masked_image.setTo(cv::Scalar(0,0,0));
            color_image_bgr.copyTo(masked_image, mask); // Only areas where mask is 255 will be visible

            // Enhance contrast
            masked_image = enhance_contrast(masked_image);

            // Compute SIFT features
            std::vector<cv::KeyPoint> kp; 
            cv::Mat desc;
            compute_sift_features(masked_image, kp, desc);

            // This part calculates an initial guess of rotation and translation matrices
            if (counter>0){
                auto feature = features.back();
                std::vector<cv::KeyPoint> kp_prev = feature.first;
                cv::Mat desc_prev = feature.second;

                std::vector<cv::DMatch> matches = match_features(desc_prev, desc);

                if (matches.size() < 5){
                    std::cout << "Not enough matches to compute pose. At least 5 matches are required." << std::endl;
                    continue;
                }

                cv::Mat depth_prev = depths.back();
                cv::Mat image_prev = images.back();

                std::vector<cv::Point3f> points1_3D, points2_3D;
                
                get_3D_points_matched(matches, kp_prev, kp, depth_prev, depth, camera_matrix, min_depth, max_depth, points1_3D, points2_3D);
            
                if (points1_3D.size() < 5){
                    std::cout << "Not enough matched points to compute pose. At least 5 matches are required." << std::endl;
                    continue;
                }
                
                matched_image = draw_matches(image_prev, kp_prev, color_image_bgr, kp, matches);

                cv::Mat R, t;
                estimate_camera_pose(kp_prev, kp, matches, camera_matrix, R, t);

                compute_scale_and_transform(R, t, points1_3D, points2_3D);

                // Convert rotation matrix to rotation vector
                cv::Mat rotation_vector;
                cv::Rodrigues(R, rotation_vector);

                // Calculate the angle of rotation (in radians)
                double angle = cv::norm(rotation_vector);

                // Normalize rotation vector to get the axis of rotation
                cv::Mat axis = rotation_vector / angle;

                // Convert angle to degrees (optional)
                double angle_degrees = angle * (180.0 / CV_PI);

                // Transform cv matrices to Eigen
                //Eigen::Matrix3d R_eigen;
                cv::cv2eigen(R, R_eigen);

                //Eigen::Vector3d t_eigen;
                cv::cv2eigen(t, t_eigen);

                std::cout << "Rotation: " << axis << std::endl;
                std::cout << "Angle in degrees: " << angle_degrees << std::endl;
                std::cout << "Translation: " << t_eigen << std::endl;

                matched_images.push_back(matched_image.clone());
                rotations.push_back(R_eigen);
                translations.push_back(t_eigen);
            
            }

            images.push_back(color_image_bgr.clone());
            images_segmented.push_back(masked_image.clone());
            depths.push_back(depth.clone());
            features.push_back(std::make_pair(kp, desc));

            // Create a point cloud object
            pc.map_to(color_frame);  // Map the point cloud to the color frame

            // Generate point cloud from depth frame
            rs2::points points = pc.calculate(depth_frame);

            // Get vertex and texture coordinate data
            auto vertices = points.get_vertices();  // 3D coordinates
            auto tex_coords = points.get_texture_coordinates();  // Texture coordinates
            // Get the number of points in the point cloud (size of the arrays)
            size_t num_points = points.size();

            // Filtered output
            std::vector<Eigen::Vector3d> filtered_vertices;
            std::vector<Eigen::Vector3d> filtered_colors;

            // Loop over the point cloud from the camera
            for (size_t i = 0; i < num_points; ++i) {
                // Directly access the vertex and texture coordinates
                float x = vertices[i].x;
                float y = vertices[i].y;
                float z = vertices[i].z;

                // Keep points within the clipping range
                if (z >= min_depth && z <= max_depth) {
                    float u = tex_coords[i].u;
                    float v = tex_coords[i].v;

                    // Check if texture coordinates are valid
                    if (u >= 0 && u <= 1 && v >= 0 && v <= 1) {
                        int x_img = static_cast<int>(u * color_image.cols);
                        int y_img = static_cast<int>(v * color_image.rows);

                        // Filter the points based on the mask
                        if (mask.at<uchar>(y_img, x_img) > 0) {  // Ensure mask is nonzero
                            filtered_vertices.push_back(Eigen::Vector3d(x, y, z));

                            cv::Vec3b color = color_image.at<cv::Vec3b>(y_img, x_img);
                            filtered_colors.push_back(Eigen::Vector3d(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0));  // Normalize to [0,1]
                        }
                    }
                }
            }

            // Create Open3D point cloud
            auto point_cloud = std::make_shared<open3d::geometry::PointCloud>();
            point_cloud->points_ = filtered_vertices;
            point_cloud->colors_ = filtered_colors;

            // Save point cloud to PLY
            std::ostringstream ss;
            ss << output_dir << "/" << pc_dir << "/pointclouds_" << std::setw(2) << std::setfill('0') << counter << ".ply";
            open3d::io::WritePointCloud(ss.str(), *point_cloud, true);


            // Visualization part
            // If the counter is 0, the current pointcloud is the first pointcloud
            if (counter==0){
                plot_rotations.push_back(Eigen::Matrix3d::Identity());
                plot_translations.push_back(Eigen::Vector3d::Zero());
                plot_pc.push_back(preprocess_pcd(*point_cloud, voxel_size));
                merged_pc += *plot_pc[0];
            }
            else{
                Eigen::Matrix3d init_rot = plot_rotations.back() * R_eigen.transpose();
                Eigen::Vector3d init_trans = plot_translations.back() - (plot_rotations.back() * R_eigen.transpose() * t_eigen);

                // Preprocess the point cloud
                auto init_pc = *preprocess_pcd(*point_cloud, voxel_size);

                // Perform ICP to refine the transformation
                auto [new_pc, transformation_icp, information_icp] = run_icp(init_pc, merged_pc, init_rot, init_trans, coarse_threshold, fine_threshold);
                merged_pc += *new_pc;
                
                // new_pc = transform_pointcloud(init_pc, init_rot, init_trans);

                // Extract rotation and translation from transformation matrix
                Eigen::Matrix3d new_rot = transformation_icp.block<3,3>(0, 0);
                Eigen::Vector3d new_trans = transformation_icp.block<3,1>(0, 3);

                std::cout << "Final rot: " << new_rot << std::endl;
                std::cout << "Final trans: " << new_trans << std::endl;

                // Append the new point cloud, rotation, and translation for the next iteration
                plot_pc.push_back(new_pc);
                plot_rotations.push_back(new_rot);
                plot_translations.push_back(new_trans);

            }

            // vis.ClearGeometries();
            vis.AddGeometry(plot_pc.back()); //std::make_shared<open3d::geometry::PointCloud>(*preprocessed_pc));

            counter++;

        }
    }

    // Cleanup
    vis.DestroyVisualizerWindow();
    return 0;
}