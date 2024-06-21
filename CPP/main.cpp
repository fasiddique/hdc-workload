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

    //  
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
 * @brief Test function for the HDC class.
 */
bool train_test() {

    // TODO: avoid hardcoding 
    int N_DIM = 2048;
    bool BINARY = false;
    // Initialize inputs for testing
    int n_class = 5; 
    int n_lv = 21; 

    // Create a Dataset object
    Dataset dataset;

    int n_id = dataset.sample_size;
    
    if (dataset.load_dataset() != 0) {
        std::cerr << "Failed to load the dataset" << std::endl;
        return true;
    }

    // Print dataset parameters
    std::cout << "Test Size: " << dataset.test_size << std::endl;
    std::cout << "Train Size: " << dataset.train_size << std::endl;
    std::cout << "Sample Size: " << dataset.sample_size << std::endl;

    auto ds_train = dataset.get_trainset();
    auto ds_test = dataset.get_testset();

    // HDC Model
    HDC hdc_model(n_class, n_lv, n_id, N_DIM, BINARY);

    // HDC Encoding Step
    std::vector<std::vector<int>> train_enc = hdc_model.encode(ds_train.first);

    std::vector<std::vector<int>> test_enc = hdc_model.encode(ds_test.first);

    // Init. Training
    hdc_model.train_init(train_enc, ds_train.second);

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
