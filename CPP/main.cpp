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

    std::string dataset_name("EMG_Hand");
    if (dataset.load_dataset(dataset_name) != 0) {
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
 * @brief Read the HDC parameters from the file.
 */
bool open_hdc_parameters(std::string dataset_name, int& n_dim, bool& binary, int& train_epochs, int& n_lv, int& n_class) {
    std::string filename = "./dataset/" + dataset_name + "/hdc_parameters";
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return true;
    }

    std::string line;
    
    std::getline(file, line);
    n_dim = std::stoi(line);

    
    std::getline(file, line);
    binary = std::stoi(line);



    std::getline(file, line);
    train_epochs = std::stoi(line);

    

    std::getline(file, line);
    n_lv = std::stoi(line);

    

    std::getline(file, line);
    n_class = std::stoi(line);

    
    return false;
    
}

/**
 * @brief Test function for the HDC class.
 */
bool train_test(std::string& dataset_name) {

    // TODO: avoid hardcoding 
    int n_dim = 2048;
    bool binary = false;
    // Initialize inputs for testing
    int n_class = 5; 
    int n_lv = 21; 
    int train_epochs = 20;
    

    // Create a Dataset object
    Dataset dataset;

    if (open_hdc_parameters(dataset_name, n_dim, binary, train_epochs, n_lv, n_class)) {
        return true;
    }

    std::cout << "INFO: n_dim = " << n_dim << std::endl;
    std::cout << "INFO: binary = " << binary << std::endl;
    std::cout << "INFO: n_class = " << n_class << std::endl;
    std::cout << "INFO: n_lv = " << n_lv << std::endl; 
    std::cout << "INFO: train_epochs = " << train_epochs << std::endl;

    if (dataset.load_dataset(dataset_name) != 0) {
        std::cerr << "Failed to load the dataset" << std::endl;
        return true;
    }

    int n_id = dataset.sample_size;
    
    
    // Print dataset parameters
    std::cout << "INFO: Test Size: " << dataset.test_size << std::endl;
    std::cout << "INFO: Train Size: " << dataset.train_size << std::endl;
    std::cout << "INFO: Sample Size: " << dataset.sample_size << std::endl;

    auto ds_train = dataset.get_trainset();
    auto ds_test = dataset.get_testset();

    
    // HDC Model
    HDC hdc_model(n_class, n_lv, n_id, n_dim, binary);

    // HDC Encoding Step
    std::vector<std::vector<int>> train_enc = hdc_model.encode(ds_train.first);

    std::vector<std::vector<int>> test_enc = hdc_model.encode(ds_test.first);

    // Init. Training
    hdc_model.train_init(train_enc, ds_train.second);

    // Initial test accuracy
    double test_acc = hdc_model.test(test_enc, ds_test.second);
    std::cout << "Init. test acc. is " << test_acc << std::endl;


    
    // Re-training
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





int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_name>" << std::endl;
        return 1;
    }

    std::string dataset_name(argv[1]);
    bool result = train_test(dataset_name);

    if (result) {
        std::cerr << "Test failed." << std::endl;
        return 1;
    }

    std::cout << "Test passed." << std::endl;
    return 0;
}
