import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Dilates the mask slightly to facilitate matching
def dilate_mask(mask, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

# Erodes the mask slightly to facilitate matching
def erode_mask(mask, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1)

# Function that enhances the contrast of the image to improve the quality of SIFT matching
def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Apply CLAHE
    l = clahe.apply(l)
    enhanced_image = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)

# Function that applies a mask to an image
def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask.astype(np.uint8) * 255)  # Mask should be 0 or 255

# Function that computes sift features
def compute_sift_features(image):
    sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=10)  # Adjust thresholds
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# Function that performs brute force matching
def match_features(desc1, desc2):
    # # Initialize FLANN-based matcher
    # index_params = dict(algorithm=1, trees=5)  # KDTree
    # search_params = dict(checks=50)  # Checks
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # # Match descriptors
    # matches = flann.knnMatch(desc1, desc2, k=2)  # k=2 for Lowe's ratio test

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1,desc2,k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # Ratio threshold
            good_matches.append(m)
    return good_matches

# This function gets the 3D points that have been matched in 2D. They also have to be within the desired depth range
def get_3d_points_matched(matches, keypoints1, keypoints2, depth1, depth2, camera_matrix, min_depth, max_depth):

    # Intrinsic parameters
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # Arrays that will hold teh final points
    points1_3D = []
    points2_3D = []

    # Loop over all the matches
    for match in matches:

        # Get the points matched in both images, in integer format
        u1, v1 = keypoints1[match.queryIdx].pt
        u2, v2 = keypoints2[match.trainIdx].pt
        u1, v1, u2, v2 = int(round(u1)), int(round(v1)), int(round(u2)), int(round(v2))
        
        # Get the depth using the depthmaps
        Z1 = depth1[v1, u1]
        Z2 = depth2[v2, u2]

        # If the depth is outside the limit, abort
        if Z1<min_depth or Z1>max_depth or Z2<min_depth or Z2>max_depth:
            continue

        # Calculate the 3D coordinates from the intrinsic parameters and the 2D coordinates
        X1 = (u1 - cx) * Z1 / fx
        Y1 = (v1 - cy) * Z1 / fy
        X2 = (u2 - cx) * Z2 / fx
        Y2 = (v2 - cy) * Z2 / fy

        # Append to the arrays
        points1_3D.append([X1, Y1, Z1])
        points2_3D.append([X2, Y2, Z2])

    return  np.array(points1_3D), np.array(points2_3D)

# Function that returns the image with matches
def draw_matches(image1, kp1, image2, kp2, matches):
    matched_image = cv2.drawMatches(image1, kp1, image2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image

# Function that calculates the rotation and translation (THE LATTER WITHOUT SCALE) given the matched points and the camera matrix
def estimate_camera_pose(kp1, kp2, matches, camera_matrix):
    
    # Convert keypoints to points
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Compute essential matrix
    E, mask = cv2.findEssentialMat(points1, points2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Decompose essential matrix to get pose
    _, R, t, _ = cv2.recoverPose(E, points1, points2, camera_matrix)
    return R, t