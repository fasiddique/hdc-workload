#include <iostream> 
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include "dataset.h"
#include "utils.h"
#include "hdc.h"


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

bool test_encode() {
    // Initialize inputs for testing 
    int n_data = 1; // 1735
    int n_class = 5;
    int n_lv = 21;
    int n_id = 1024;
    int N_DIM = 2048;
    bool BINARY = false;

    int fixed_value = 0;

    // Define the shape
    std::vector<std::vector<int>> fixed_value_tensor(n_data, std::vector<int>(n_id, fixed_value));

    // HDC Model
    HDC hdc_model(n_class, n_lv, n_id, N_DIM, BINARY);

    // HDC Encoding Step
    std::vector<std::vector<int>> fixed_value_tensor_enc = hdc_model.encode(fixed_value_tensor);

    // Print encoded tensor shape
    std::cout << "[DEBUG] fixed_value_tensor_enc.size() = " << fixed_value_tensor_enc.size() << " x " << fixed_value_tensor_enc[0].size() << std::endl;

    // Print encoded tensor
    print_2d_vector(fixed_value_tensor_enc);

    // Save to a file
    std::ofstream outfile("tensor_data.bin", std::ios::binary);
    if (!outfile) {
        std::cerr << "Error opening file for writing" << std::endl;
        return true;
    }

    for (const auto& vec : fixed_value_tensor_enc) {
        outfile.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(int));
    }

    outfile.close();
    return false;
}


/**
 * @brief Fills a vector with a fixed value.
 * @param n Number of vectors to generate.
 * @param m Size of each vector.
 * @param value Fixed value to fill the vectors.
 * @return Filled vector of vectors.
 */
std::vector<std::vector<int>> fill_vector(size_t n, size_t m, int value) {
    return std::vector<std::vector<int>>(n, std::vector<int>(m, value));
}

/**
 * @brief Test function for the HDC class.
 */
bool train_test() {
    // Initialize inputs for testing
    size_t n_train_data = 1; // 1735
    size_t n_test_data = 1; // 579
    int n_class = 5;
    int n_lv = 5; // 21
    int n_id = 32; // 1024
    int N_DIM = 64; // 2048
    bool BINARY = false;

    int train_fixed_value = 0;
    int test_fixed_value = 0;

    auto ds_train = std::make_pair(fill_vector(n_train_data, n_id, train_fixed_value),
                                   std::vector<int>(n_train_data, train_fixed_value));

    auto ds_test = std::make_pair(fill_vector(n_test_data, n_id, test_fixed_value),
                                  std::vector<int>(n_test_data, test_fixed_value));

    // HDC Model
    HDC hdc_model(n_class, n_lv, n_id, N_DIM, BINARY);

    // HDC Encoding Step
    std::vector<std::vector<int>> train_enc = hdc_model.encode(ds_train.first);
    std::vector<std::vector<int>> test_enc = hdc_model.encode(ds_test.first);

    // Init. Training
    hdc_model.train_init(train_enc, ds_train.second);

    // Print class hypervectors
    const auto& class_hvs = hdc_model.get_class_hvs();
    for (const auto& hv : class_hvs) {
        for (const auto& val : hv) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // Initial test accuracy
    double test_acc = hdc_model.test(test_enc, ds_test.second);
    std::cout << "Init. test acc. is " << test_acc << std::endl;


    // Re-training
    int train_epochs = 20;
    int val_epochs = 5;

    for (int i = 0; i < train_epochs; ++i) {
        hdc_model.train(train_enc, ds_train.second);

        if ((i + 1) % val_epochs == 0) {
            test_acc = hdc_model.test(test_enc, ds_test.second);
            std::cout << "Test acc. @ epoch " << (i + 1) << "/" << train_epochs << " is " << test_acc << std::endl;
        }
    }

    test_acc = hdc_model.test(test_enc, ds_test.second);
    std::cout << "Final test acc. is " << test_acc << std::endl;

    // if (BINARY) {
    //     for (auto& hv : hdc_model.get_class_hvs()) {
    //         hv = binarize(hv);
    //     }
    // }

    return false;
}





int main() {
    bool result = train_test();
    if (result) {
        std::cerr << "Test failed." << std::endl;
        return 1;
    }
    std::cout << "Test passed." << std::endl;
    return 0;
}
