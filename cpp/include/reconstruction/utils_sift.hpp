#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <iostream>

// Function to dilate the mask to facilitate matching
cv::Mat dilate_mask(const cv::Mat& mask, int kernelSize = 3) {
    cv::Mat kernel = cv::Mat::ones(kernelSize, kernelSize, CV_8U);
    cv::Mat dilatedMask;
    cv::dilate(mask, dilatedMask, kernel);
    return dilatedMask;
}

// Function to erode the mask to facilitate matching
cv::Mat erode_mask(const cv::Mat& mask, int kernelSize = 3) {
    cv::Mat kernel = cv::Mat::ones(kernelSize, kernelSize, CV_8U);
    cv::Mat erodedMask;
    cv::erode(mask, erodedMask, kernel);
    return erodedMask;
}

// Function to enhance contrast using CLAHE
cv::Mat enhance_contrast(const cv::Mat& image) {
    cv::Mat labImage, l, a, b;
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> labChannels;
    cv::split(labImage, labChannels);
    l = labChannels[0];
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(l, l);
    cv::merge(std::vector<cv::Mat>{l, labChannels[1], labChannels[2]}, labImage);
    cv::Mat enhancedImage;
    cv::cvtColor(labImage, enhancedImage, cv::COLOR_Lab2BGR);
    return enhancedImage;
}

// Function to apply a mask to an image
cv::Mat apply_mask(const cv::Mat& image, const cv::Mat& mask) {
    cv::Mat maskedImage;
    cv::bitwise_and(image, image, maskedImage, mask * 255);
    return maskedImage;
}

// Function to compute SIFT features
void compute_sift_features(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    auto sift = cv::SIFT::create(0, 3, 0.01, 10, 1.6);
    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}

// Function to perform brute-force matching with Lowe's ratio test
std::vector<cv::DMatch> match_features(const cv::Mat& desc1, const cv::Mat& desc2) {
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(desc1, desc2, knnMatches, 2);

    std::vector<cv::DMatch> good_matches;
    for (const auto& m : knnMatches) {
        if (m[0].distance < 0.75 * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }
    return good_matches;
}

// Function to compute 3D points from matches
void get_3D_points_matched(const std::vector<cv::DMatch>& matches,
                        const std::vector<cv::KeyPoint>& keypoints1,
                        const std::vector<cv::KeyPoint>& keypoints2,
                        const cv::Mat& depth1,
                        const cv::Mat& depth2,
                        const cv::Mat& cameraMatrix,
                        float minDepth,
                        float maxDepth,
                        std::vector<cv::Point3f>& points1_3D,
                        std::vector<cv::Point3f>& points2_3D) {

    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);

    for (const auto& match : matches) {
        int u1 = static_cast<int>(std::round(keypoints1[match.queryIdx].pt.x));
        int v1 = static_cast<int>(std::round(keypoints1[match.queryIdx].pt.y));
        int u2 = static_cast<int>(std::round(keypoints2[match.trainIdx].pt.x));
        int v2 = static_cast<int>(std::round(keypoints2[match.trainIdx].pt.y));

        float Z1 = depth1.at<float>(v1, u1);
        float Z2 = depth2.at<float>(v2, u2);

        if (Z1 < minDepth || Z1 > maxDepth || Z2 < minDepth || Z2 > maxDepth) {
            continue;
        }

        float X1 = (u1 - cx) * Z1 / fx;
        float Y1 = (v1 - cy) * Z1 / fy;
        float X2 = (u2 - cx) * Z2 / fx;
        float Y2 = (v2 - cy) * Z2 / fy;

        points1_3D.emplace_back(X1, Y1, Z1);
        points2_3D.emplace_back(X2, Y2, Z2);
    }
}

// Function to draw matches
cv::Mat draw_matches(const cv::Mat& image1, const std::vector<cv::KeyPoint>& kp1,
                    const cv::Mat& image2, const std::vector<cv::KeyPoint>& kp2,
                    const std::vector<cv::DMatch>& matches) {
    cv::Mat matchedImage;
    cv::drawMatches(image1, kp1, image2, kp2, matches, matchedImage, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return matchedImage;
}

// Function to estimate camera pose
void estimate_camera_pose(const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
                        const std::vector<cv::DMatch>& matches, const cv::Mat& cameraMatrix,
                        cv::Mat& R, cv::Mat& t) {
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : matches) {
        points1.push_back(kp1[match.queryIdx].pt);
        points2.push_back(kp2[match.trainIdx].pt);
    }

    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(points1, points2, cameraMatrix, cv::RANSAC, 0.999, 1.0, mask);
    cv::recoverPose(E, points1, points2, cameraMatrix, R, t, mask);
}

void compute_scale_and_transform(const cv::Mat& R, cv::Mat& t, 
                              const std::vector<cv::Point3f>& points1_3D, 
                              const std::vector<cv::Point3f>& points2_3D) {
    std::vector<float> distances;

    for (size_t i = 0; i < points1_3D.size(); i++) {
        // Rotate points1_3D to align with second set
        cv::Mat pt1 = (cv::Mat_<double>(3,1) << points1_3D[i].x, points1_3D[i].y, points1_3D[i].z);
        cv::Mat pt1_rotated = R * pt1;

        // Compute Euclidean distance between transformed point and actual second point
        cv::Point3f pt1_rot(pt1_rotated.at<double>(0,0), pt1_rotated.at<double>(1,0), pt1_rotated.at<double>(2,0));
        float distance = cv::norm(points2_3D[i] - pt1_rot);
        distances.push_back(distance);
    }

    // Compute median distance (scale)
    std::nth_element(distances.begin(), distances.begin() + distances.size() / 2, distances.end());
    float scale = distances[distances.size() / 2];

    // Scale translation vector
    t *= scale;
}