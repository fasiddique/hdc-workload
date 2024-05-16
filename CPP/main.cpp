#include <iostream> 
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include "dataset.h"

/**
 * @brief Test function to demonstrate the usage of the Dataset class.
 * 
 * This function creates an instance of the Dataset class, loads the dataset,
 * prints some sample data and labels from the training and test sets, and
 * calculates the checksum of the dataset.
 * 
 * @return false if the dataset was loaded successfully, true otherwise.
 */
bool test_dataset() {
    // Create a Dataset object
    Dataset dataset;

    // Load the dataset
    if (dataset.load_dataset() != 0) {
        std::cerr << "Failed to load the dataset" << std::endl;
        return true;
    }

    // Print dataset parameters
    std::cout << "Test Size: " << dataset.test_size << std::endl;
    std::cout << "Train Size: " << dataset.train_size << std::endl;
    std::cout << "Sample Size: " << dataset.sample_size << std::endl;

    // Print some train data values
    std::cout << "Train Data:" << std::endl;
    for (int i = 0; i < std::min(5, dataset.train.size); ++i) {
        std::cout << "Sample " << i << ": ";
        for (int j = 0; j < std::min(5, dataset.train.sample_size); ++j) {
            std::cout << dataset.train.values[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Print some train labels
    std::cout << "Train Labels:" << std::endl;
    for (int i = 0; i < std::min(5, dataset.train.size); ++i) {
        std::cout << dataset.train.labels[i] << " ";
    }
    std::cout << std::endl;

    // Print some test data values
    std::cout << "Test Data:" << std::endl;
    for (int i = 0; i < std::min(5, dataset.test.size); ++i) {
        std::cout << "Sample " << i << ": ";
        for (int j = 0; j < std::min(5, dataset.test.sample_size); ++j) {
            std::cout << dataset.test.values[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Print some test labels
    std::cout << "Test Labels:" << std::endl;
    for (int i = 0; i < std::min(5, dataset.test.size); ++i) {
        std::cout << dataset.test.labels[i] << " ";
    }
    std::cout << std::endl;

    // Compute and print the checksum
    int checksum = dataset.get_checksum();
    std::cout << "Checksum: " << checksum << std::endl;

    return false;
}


/**
 * @brief Tests reading a tensor from a binary file and stores it in a 2D vector.
 *
 * This function reads a binary file containing tensor data and stores the data in a
 * 2D vector. The tensor is assumed to have a fixed number of rows and columns.
 * The function prints the contents of the tensor to the standard output.
 *
 * @return true if there was an error opening the file, false otherwise.
 */
bool test_read_tensor() {
    // Define the dimensions of the tensor
    const int rows = 1; // n_data
    const int cols = 2048; // N_DIM (or as required)

    // Read the binary file
    std::ifstream infile("tensor_data.bin", std::ios::binary);

    // Check if the file was opened successfully
    if (!infile) {
        std::cerr << "Error opening file" << std::endl;
        return true;
    }

    // Create a 2D vector to hold the data
    std::vector<std::vector<int>> tensor_data(rows, std::vector<int>(cols));

    // Read the data from the file into the 2D vector
    for (int i = 0; i < rows; ++i) {
        infile.read(reinterpret_cast<char*>(tensor_data[i].data()), cols * sizeof(int));
    }

    // Close the file
    infile.close();

    // Example: Access the data
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << tensor_data[i][j] << " ";
        }
        std::cout << std::endl;
    }
    return false;
}




/**
 * @brief Main function to execute the test_dataset function.
 * 
 * This function calls the test_dataset function and returns its result.
 * 
 * @return 0 if the test_dataset function succeeds, 1 otherwise.
 */
int main() {
    if (test_read_tensor()) {
        return true;
    }

    return false;
}
