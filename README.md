<!--![Rotating Mesh](assets/gif_rayquaza.gif)-->

<img src="assets/gif_pikachu.gif" height="200"> <img src="assets/gif_totodile.gif" height="200"> <img src="assets/gif_rayquaza.gif" height="200"> <img src="assets/gif_camel.gif" height="200">

# 3D RECONSTRUCTION PROJECT WITH REALSENSE CAMERA

This repository contains software to perform 3D reconstruction (merged pointcloud and mesh) by using the Realsense D435 camera to take images and pointclouds thanks to the stereo capabilities of the camera. I created a Python and C++ version. The C++ version has been tested on Nvidia Jetson AGX Orin. The steps are as follows:

1. Each captured image is segmented using [SAM 2.1](https://github.com/facebookresearch/sam2) to isolate the object whose pointcloud should be extracted.

2. The camera also provides a depthmap. We convert this depthmap into a pointcloud by taking those points lying inside the mask.

3. Each pair of consecutive images are matched using SHIFT feature matching. Since we know the intrinsic parameters of the camera, and also the depth of each point (thanks to the stereo), we can calculate an initial approximation of the rotation and translation between one object and the next.

4. Once the pointclouds and the initial guesses of rotations and translation are calculated, we perform a PoseGraph optimization algorithm to build a final version of the pointcloud, taking into account loop closures to improve the overall matching. This provides a final merged pointcloud.

5. From the merged pointcloud, we create the mesh like those represented by the GIFs at the beginning of this sheet. Pretty cool, right?

These steps are represented in the following main scheme. There are also various substeps, and I will highlight some of the most important, specially if they help making proper use of the programs provided.

<img src="assets/scheme.gif">

# REQUIREMENTS

The main common libraries to install for both the Python and C++ versions are the `open_3d`, `opencv` and `realsense2` libraries. However, the segmentation part is slightly different:

- In Python, the `ultralytics` framework is employed, which also supports the SAM 2.1 model. This model accepts an image and a variable set of positive and negative points to better guide the segmentation.

- In C++, I employ my previous developed [TensorRT](https://github.com/Raessan/tensorrt_lib) library, which is adapted for Jetson Devices. This provides a faster inference than the Python version (and, in general, the whole program runs faster in C++). However, creating the required ONNX model(s) is a bit more involved. I provide inside the folder `onnx_export` a notebook to export the SAM model to ONNX. It is based on [this file](https://github.com/shubham0204/Segment-Anything-Android/blob/main/notebooks/SAM2_ONNX_Export.ipynb), but I performed slight modifications. Mainly, I had problems with leaving the dynamic_axes in the decoder, so I decided to set the variables constant for the TensorRT inference. For that reason, the number of points that can be used to guide the segmentation is constant (in the file I proposed, 5 points, but the model can be re-created for more or less points). The opset version has also been changed to match my TensorRT version. The program outputs two ONNX models: the encoder and the decoder. Also, tu run `onnx_export` the original [SAM 2 repository](https://github.com/facebookresearch/sam2) has to be installed. I recommend doing this export in a different Conda environment because it requires specific versions of libraries like `torch`.

For the C++ version, remember to build the libraries by positioning in the cpp folder and running:

```bash
mkdir build
cd build
cmake ..
make
```

# USAGE OF THE PROGRAMS

The C++ and Python versions work exactly the same, and the programs are in order: `01_get_samples`, `02_reconstruction` and `03_create_mesh`. For Python, get in the `python` folder and run: `python 01_get_samples.py`. For C++, get in the `build` folder and run `./01_get_samples`.

## 01_get_samples

This program collects the pointclouds and initial guesses of rotation/translation. It builds an interface to see how the process is going, based on two windows:

1. A camera-based window where we can see the images before we take them and configure the segmentation.
2. An Open3D window to see how the pointcloud reconstruction is working. THIS IS NOT THE DEFINITIVE RECONSTRUCTION, but just a guess to see if the process is going well. The final reconstruction is provided in `02_reconstruction`.

The camera-based window has three images: the real-time data capture, the segmentation, and the SIFT matching. We can interact with the first image by adding positive points (left click) or negative points (right click) for the segmentation. By default, the central pixel of the image is always positive, so the object should be placed around it. In the Python version, the number of positive and negative points are ilimited. But in the C++ version, this number is limited and the variable `n_points` should have the same value as that of the `sam_onnx_exporter` (currently, 5). If more clicks are done, they are ignored. There are three ways to interact with this window (which should have the focus in order to obey our keypress events):

1. Space key: Activates the whole mechanism. Captures the image, calculates the mask (using also the positive and negative points), creates the pointcloud using the depthmap capabilities of Realsense and extracts the SIFT features. If it is the first image, nothing else is done. Otherwise, the image is compared with the previous one to match their SIFT features, obtain the essential matrix from the matches, and from it the rotation and translation (warning: the translation directly calculated using `cv2.findEssentialMat` provides a unitary (unscaled) vector for the translation, so it is scaled afterwards using the depthmap). The masked and matched images are plotted.

To provide the initial guess for the reconstruction, an initial ICP registration is performed. For this, we have accumulative rotations and rotations which take all the pointclouds to the reference frame of the first pointcloud, and each new rotation and translation obtained from the matching is refined using ICP. This is updated in the Open3D window, which can be zoomed, panned, etc. The ICP is not of great quality because we are not computing normals (otherwise it would take too long) and are using PointToPoint ICP, so the object is expected to not have a good loop closure at this stage.

2. `r` key: This key deletes the last sample collected (including image, pointcloud, features, etc.), as if it hadn't been taken. This is useful if we see that the recently added pointcloud is too far from the previous. This improves the step-by-step experience of creating valid samples without having to start over.

3. `d` key: It resets the positive and negative points to perform the segmentation of the current sample (only the central blue point will remain after pressing this key), but it does not delete any previous sample.

4. `q` key: It finishes the data collection and saves the variables, so it is now ready for the final reconstruction. The result has to be a folder with N pointclouds and N-1 rotations and translations.

## 02_reconstruction

This program performs PoseGraph optimization on the previously generated pointclouds. The variable `n_merge` controls how many of the pointclouds are used (is set to 0, all the pointclouds are registered). It performs PoseGraph optimization to better handle the loop closures, the refined ICP now uses PointToPlane to improve the matching. The result should be more satisfactory than the incremental ICP from the previous script. The merged final pointcloud is saved. Feel free to modify some of the parameters of the PoseGraph optimization. I selected some that worked for me to generate the meshes in the given GIFs.

## 03_create_mesh

From the previously generated merged pointcloud, it generates a mesh. Here I should comment about the variable `depth`. Ideally, the higher value, the more precise the mesh will be (around 9 should be pretty good). However, in some devices I found that a high `depth` value was causing the mesh operation to fail. In other devices, it worked for me with `depth=9`. It's not clear to me if it is related to the hardware itself or the version of the libraries.