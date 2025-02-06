import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import os
import re
from ultralytics import SAM
from utils.utils_sift import *
from utils.utils_pointcloud import *
import time

# Set up output directory
OUTPUT_DIR = "../data/data_pikachu"
MODEL_DIR = "../models"

# Directories for the pointcloud, rotation matrices and translation vectors
PC_DIR = "pointclouds"
ROTATIONS_DIR = "rotations"
TRANSLATIONS_DIR = "translations"

# Define a clipping range (e.g., keep points between 0.1m and 1.0m)
MIN_DEPTH = 0.1
MAX_DEPTH = 1.0
# Width of the image
WIDTH = 640
# Height of the image
HEIGHT = 480
# Width of the pointcloud plot with Open3D
WIDTH_PC_PLOT = 800
# Height of the pointcloud plot with Open3D
HEIGHT_PC_PLOT = 600
# Voxel size of the pointcloud
VOXEL_SIZE = 0.001
# Coarse threshold for ICP
COARSE_THRESHOLD = VOXEL_SIZE*15
# Fine threshold for ICP
FINE_THRESHOLD = VOXEL_SIZE*5

# Creation of directories
os.makedirs(os.path.join(OUTPUT_DIR, PC_DIR), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, ROTATIONS_DIR), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, TRANSLATIONS_DIR), exist_ok=True)
os.makedirs(os.path.join(MODEL_DIR), exist_ok=True)

# Load a model
model = SAM(os.path.join(MODEL_DIR,"sam2.1_s.pt"))

# Initialize the pipeline for RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)

# Start the pipeline
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Create a pointcloud object
pc = rs.pointcloud()
# Create an align object to align depth to color frame
align = rs.align(rs.stream.color)

print("Press the space key to capture an image and point cloud.")
print("Press 'r' to remove previous sample.")
print("Press 'd' to delete auxiliar points.")
print("Press 'q' to quit the app.")


# Image that will be displayed
complete_image = np.zeros((HEIGHT*2, WIDTH*2, 3), dtype=np.uint8)

# Segmented image and image with matches
masked_image = None
matched_image = None

# Counter
counter = 0

# Auxiliar vectors needed
images = []
matched_images = []
images_segmented = []
masks = []
rotations = []
translations = []
depths = []
features = []

pc_plot = []
rot_plot = []
trans_plot = []


prev_frame_number = -1
stuck_frames = 0

# Positive and negative points for the SAM model
positive_points = [[WIDTH//2, HEIGHT//2]]
negative_points = []

# Merged pointcloud
merged_pc = o3d.geometry.PointCloud()

# Visualizer
vis = initialize_visualizer(WIDTH_PC_PLOT, HEIGHT_PC_PLOT, "Pointclouds")

# Function to handle mouse events
def manage_clicks(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:  # Left button click
        positive_points.append([x, y])  # Add to positive_points
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right button click
        negative_points.append([x, y])  # Add to negative_points

try:

    while True:

        vis.poll_events()
        vis.update_renderer()

        # Wait for a new set of frames
        frames = pipeline.wait_for_frames()
        # Align the frames
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        frame_number = color_frame.get_frame_number()

        if not depth_frame or not color_frame:
            continue

        if frame_number == prev_frame_number:
            stuck_frames += 1
        else:
            stuck_frames = 0

        prev_frame_number = frame_number

        if stuck_frames > 5:  # Threshold for stalled pipeline
            print("Pipeline seems stuck. Restarting...")
            pipeline.stop()
            pipeline.start(config)
            stuck_frames = 0

        # Retrieve the stream profile for color
        color_stream = color_frame.get_profile()
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        camera_matrix = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx], [0, color_intrinsics.fy, color_intrinsics.ppy], [0, 0, 1]])

        

        # Convert the color frame to a numpy array
        color_image_bgr = np.asanyarray(color_frame.get_data())
        # Put a circle in the middle to indicate where the object should lie
        color_image_bgr_annotated = color_image_bgr.copy()
        for point in positive_points:
            cv2.circle(color_image_bgr_annotated, point, 5, (255, 0, 0), -1)
        for point in negative_points:
            cv2.circle(color_image_bgr_annotated, point, 5, (0, 0, 255), -1)


        if counter > 0 and masked_image is not None:
            complete_image[0:HEIGHT, WIDTH:WIDTH*2, :] = masked_image
        else:
            complete_image[0:HEIGHT, WIDTH:WIDTH*2, :] = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        if counter > 1 and matched_image is not None:
            complete_image[HEIGHT:, 0:2*WIDTH, :] = matched_image
        else:
            complete_image[HEIGHT:, 0:2*WIDTH, :] = np.zeros((HEIGHT, WIDTH*2, 3), dtype=np.uint8)
            
        complete_image[0:HEIGHT, 0:WIDTH, :] = color_image_bgr_annotated

        # Display the color image using OpenCV
        cv2.imshow("Complete_image", complete_image)

        # Set mouse callback to handle drawing
        cv2.setMouseCallback("Complete_image", manage_clicks)

        depth = np.asanyarray(depth_frame.get_data()) * depth_scale # Depth in meters

        # Convert from BGR to RGB
        color_image = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)

        # Check for key press
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Pressed q!")

            # Save the rotation and translation matrices, which are the only ones needed for the final Pointcloud registration
            for i in range(len(rotations)):
                #np.save(os.path.join(OUTPUT_DIR, ROTATIONS_DIR, "rotation_%02d.npy" % i), rotations[i])
                np.savetxt(os.path.join(OUTPUT_DIR, ROTATIONS_DIR, "rotation_%02d.txt" % i), rotations[i], fmt='%.5f')  # Save with 5 decimal precision
            for i in range(len(translations)):
                #np.save(os.path.join(OUTPUT_DIR, TRANSLATIONS_DIR, "translation_%02d.npy" % i), translations[i])
                np.savetxt(os.path.join(OUTPUT_DIR, TRANSLATIONS_DIR, "translation_%02d.txt" % i), translations[i], fmt='%.5f')
            break
            
        # Key "r" will delete the last generated sample
        elif key == ord('r'):

            # Set positive and negative points to their original value
            positive_points = [[WIDTH//2, HEIGHT//2]]
            negative_points = []

            # If counter >0 we can delete most of the variables
            if counter>0:
                # Delete last element of vectors
                images.pop()
                depths.pop()
                images_segmented.pop()
                masks.pop()                
                features.pop()
                pc_plot.pop()
                rot_plot.pop()
                trans_plot.pop()
                # Delete pointcloud
                full_path_pc = os.path.join(OUTPUT_DIR, PC_DIR, "pointcloud_%02d.ply" % (counter-1))
                os.path.join(OUTPUT_DIR,PC_DIR,"pointcloud_%02d.ply" % counter)
                if os.path.exists(full_path_pc):
                    os.remove(full_path_pc)
                else:
                    print(f"File does not exist: {full_path_pc}")

                # Redefine the merged_pc
                merged_pc.clear()
                for pointcloud in pc_plot:
                    merged_pc += pointcloud
                
                # Send the vector of pointclouds to plot to the visualizer
                vis.clear_geometries()
                vis.add_geometry(merged_pc)

            # If the counter is >1, we can also delete the matched images, rotations and translations
            if counter>1:
                matched_images.pop()
                rotations.pop()
                translations.pop()

            # Update counter number
            if counter > 0:
                counter -= 1

            # Update visualization variables
            if counter>0:
                masked_image = images_segmented[-1]
            else:
                masked_image = None
            if counter>1:
                matched_image = matched_images[-1]
            else: 
                matched_image = None

        # The key "d" just resets the positive and negative points to their original value
        elif key == ord('d'):

            positive_points = [[WIDTH//2, HEIGHT//2]]
            negative_points = []
            

        elif key == 32:  # Space key
            
            print("Pressed space!")
            # Perform segmentation on image
            merged_points = []
            labels = []
            # Adding positive points and corresponding labels
            for point in positive_points:
                merged_points.append(point)
                labels.append(1)  # Label for positive points is 1
            # Adding negative points and corresponding labels
            for point in negative_points:
                merged_points.append(point)
                labels.append(0)  # Label for negative points is 0

            results = model(color_image_bgr, points=[merged_points], labels = [labels], show=False)

            # Restore positive and negative points to original values
            positive_points = [[WIDTH//2, HEIGHT//2]]
            negative_points = []

            # Check if there are masks in the results
            if results[0].masks is None:
                print ("The SAM model couldn't provide a mask.")
                continue

            # Assuming `results` is your model's output
            mask = results[0].masks.data[0].cpu().numpy().squeeze()  # Get the Masks object

            # Scale the mask to 255 (binary mask: 0 or 255)
            mask_array = mask.astype(np.uint8) * 255  # Convert to NumPy

            # Create a colored mask (adjust RGB values as needed)
            mask_img = np.zeros_like(color_image_bgr)
            mask_img[:, :, 0] = mask_array

            # Perform SIFT feature extraction
            #dilated_mask = dilate_mask(mask, kernel_size=5)
            mask = erode_mask(mask, 3)
            masked_image = apply_mask(color_image_bgr, mask)
            masked_image = enhance_contrast(masked_image)
            
            # Compute SIFT features
            keypoints, descriptors = compute_sift_features(masked_image)
            
            # This part calculates an initial guess of rotation and translation matrices
            if counter > 0:

                # Keypoints and descriptors to compare
                kp1, desc1 = features[-1]
                kp2, desc2 = keypoints, descriptors

                # Match the features
                matches = match_features(desc1, desc2)

                # Abort if the length of matches if <5
                if len(matches) < 5:
                    print("Not enough matches to compute pose. At least 5 matches are required.")
                    continue

                # Auxiliar variables from previous iteration
                depth_ant = depths[-1]
                image_ant = images[-1]
                mask_ant = masks[-1]

                # We get the matched points in 3D. They are filtered to be between z_min and z_max
                points1_3D, points2_3D = get_3d_points_matched(matches, kp1, kp2, depth_ant, depth, camera_matrix, MIN_DEPTH, MAX_DEPTH)

                # If the matched points are less than 5, once again abort the iteration
                if len(points1_3D) < 5:
                    print("Not enough matched points to compute pose. At least 5 matches are required.")
                    continue

                # Obtain the image with matches for plotting
                matched_image = draw_matches(image_ant, kp1, color_image_bgr, kp2, matches)
                
                # Using the keypoints, matches and camera matrix, we can estimate the initial rotation and translation (THE TRANSLATION IS UNITARY VECTOR. NOT SCALED!!)
                R, t = estimate_camera_pose(kp1, kp2, matches, camera_matrix)

                # NOW WE NEED TO CALCULATE THE SCALE FOR THE TRANSLATION VECTOR
                
                # Extract 3D points and compute scale                         
                # Apply rotation to first 3D points to align them with the second set
                points1_3D_rotated = np.dot(R, points1_3D.T).T

                # Calculate the distance between the estimated and real points
                distances = np.linalg.norm(points2_3D - points1_3D_rotated, axis=1)
                
                # Set the scale as the distance
                scale = np.median(distances)
                
                # Scale translation vector
                t_scaled = t * scale

                # Convert rotation matrix to rotation vector
                rotation_vector, _ = cv2.Rodrigues(R)

                # Calculate angle of rotation (in radians)
                angle = np.linalg.norm(rotation_vector)

                # Normalize rotation vector to get the axis of rotation
                axis = rotation_vector / angle

                # Convert angle to degrees (optional)
                angle_degrees = np.degrees(angle)

                print("Rotation: ", axis)
                print("Angle in degrees: ", angle_degrees)
                print("Translation: ", t_scaled)

                # Save vectors
                matched_images.append(matched_image)
                rotations.append(R)
                translations.append(t_scaled)

            images.append(color_image_bgr)
            masks.append(mask)
            images_segmented.append(masked_image)
            depths.append(depth)
            features.append([keypoints, descriptors])

            # Map the point cloud to the color frame
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)

            # Get the vertices (3D points) and texture coordinates (mapped colors)
            vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            tex_coords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

            # Filter points based on the depth range
            filtered_vertices = []
            filtered_colors = []

            # Now loop over the pointcloud from the camera
            for i, vertex in enumerate(vertices):
                x, y, z = vertex
                if MIN_DEPTH <= z <= MAX_DEPTH:  # Only keep points within the clipping range
                    # Get the corresponding color for each vertex using texture coordinates
                    u, v = tex_coords[i]
                    if 0 <= u <= 1 and 0 <= v <= 1:  # Check if the coordinates are valid
                        x_img = int(u * color_image.shape[1])
                        y_img = int(v * color_image.shape[0])

                        # We filter the points by belonging to the mask
                        if mask[y_img, x_img]:

                            filtered_vertices.append(vertex)
                            color = color_image[y_img, x_img] / 255.0  # Normalize to [0, 1]
                            filtered_colors.append(color)

            # Convert filtered lists to numpy arrays
            filtered_vertices = np.array(filtered_vertices)
            filtered_colors = np.array(filtered_colors)

            # Create Open3D point cloud
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(filtered_vertices)
            point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

            # Save pointcloud to PLY
            o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR,PC_DIR,"pointcloud_%02d.ply" % counter), point_cloud, write_ascii=True)

            # VISUALIZATION PART
            # If the counter is 0, the current pointcloud is the first pointcloud
            if counter == 0:
                rot_plot.append(np.eye(3))
                trans_plot.append(np.zeros(3))
                pc_plot.append(preprocess_pcd(point_cloud, VOXEL_SIZE))
                merged_pc += pc_plot[0]
            #If the counter is >0, we bring the current pointcloud to the frame reference of the first pointcloud, and then perform ICP
            else:
                # This is the initial guess of rotation, using the current rotation R and the accumulated rotation of rot_plot
                init_rot = np.dot(rot_plot[-1], R.T)
                # We do the same for the translation
                init_trans = trans_plot[-1] - (rot_plot[-1] @ R.T @ t_scaled.reshape(3, 1)).squeeze()
                # We preprocess the pointcloud
                init_pc = preprocess_pcd(point_cloud, VOXEL_SIZE)
                # We perform ICP to better approximate the rotation and translation matrices
                new_pc, transformation, _ = run_icp(init_pc, merged_pc, init_rot, init_trans, COARSE_THRESHOLD, FINE_THRESHOLD)
                new_rot = transformation[:3,:3]
                new_trans = transformation[:3, 3]
                # We define a merged pointcloud with has all the current pointclouds and we will add the new one
                merged_pc += new_pc
                # Append the pointcloud, rotation and translation matrices for the next iteration
                pc_plot.append(new_pc)
                rot_plot.append(new_rot)
                trans_plot.append(new_trans)

            # This adds the new pointcloud to the visualizer
            vis.add_geometry(pc_plot[-1])

            # Adjust the counter
            counter += 1
            
finally:
    # Stop the pipeline
    pipeline.stop()
    vis.destroy_window()
    cv2.destroyAllWindows()