#include <iostream>
#include <fstream>
#include <Eigen/Dense>

// Function to load a 3x3 rotation matrix from a .txt file
bool load_rotation(const std::string& filename, Eigen::Matrix3d& rotationMatrix) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    // Read the matrix elements into the Eigen::Matrix3d
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            file >> rotationMatrix(i, j);
            if (file.fail()) {
                std::cerr << "Error: Failed to read matrix element (" << i << ", " << j << ") from " << filename << std::endl;
                return false;
            }
        }
    }

    file.close();
    return true;
}

// Function to load a 3x1 translation vector from a .txt file
bool load_translation(const std::string& filename, Eigen::Vector3d& translationVector) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    // Read the vector elements into the Eigen::Vector3d
    for (int i = 0; i < 3; ++i) {
        file >> translationVector(i);
        if (file.fail()) {
            std::cerr << "Error: Failed to read vector element (" << i << ") from " << filename << std::endl;
            return false;
        }
    }

    file.close();
    return true;
}

// Function to save Eigen matrix/vector to a .txt file
template <typename Derived>
void save_eigen_txt(const Eigen::MatrixBase<Derived>& mat, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Define the formatting: space-separated values, no alignment
    Eigen::IOFormat CleanFormat(Eigen::FullPrecision, Eigen::DontAlignCols, " ", "\n");

    file << mat.format(CleanFormat) << std::endl;
    file.close();
}