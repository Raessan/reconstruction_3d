cmake_minimum_required(VERSION 3.16)
project(reconstruction)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)

# Including the "release" feature optimizes the performance in production
set(CMAKE_BUILD_TYPE "Release")

# Add the dependency for Eigen3 and OpenCV
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Open3D REQUIRED)
find_package(realsense2 REQUIRED)

# Include directories for Eigen3 and OpenCV
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${realsense2_INCLUDE_DIRS})

# To include the "include" folder to the path, and OpenCV
include_directories(include)

# To compile the library of TensorRT
add_subdirectory(tensorrt_lib/tensorrt_cpp/tensorrt_cpp_lib)
add_subdirectory(tensorrt_lib/tensorrt_cpp/nn_handler_lib)

# To compile the get_samples program
add_executable(01_get_samples src/01_get_samples.cpp)
target_link_libraries(01_get_samples PRIVATE nn_handler_lib ${realsense2_LIBRARY} ${OpenCV_LIBS} Open3D::Open3D)

# To compile the reconstruction program
add_executable(02_reconstruction src/02_reconstruction.cpp)
target_link_libraries(02_reconstruction PRIVATE Open3D::Open3D)

# To compile the program for reconstruction from merged PC
add_executable(03_create_mesh src/03_create_mesh.cpp)
target_link_libraries(03_create_mesh PRIVATE Open3D::Open3D)