#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <utility>

#define DATASET "EMG_Hand"

/**
 * @brief Class representing a subset of data (either training or test).
 */
class DataSubset {
public:
    int size; /**< The number of samples in the subset. */
    int sample_size; /**< The number of points in each sample. */
    std::vector<std::vector<int>> values; /**< The values of the samples. */
    std::vector<int> labels; /**< The labels of the samples. */

    /**
     * @brief Reads the sample values from a file.
     * 
     * @param filename The name of the file to read from.
     * @return True if the file was read successfully, false otherwise.
     */
    bool read_values(const std::string &filename);

    /**
     * @brief Reads the sample labels from a file.
     * 
     * @param filename The name of the file to read from.
     * @return True if the file was read successfully, false otherwise.
     */
    bool read_labels(const std::string &filename);
};

bool DataSubset::read_values(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return false;
    }

    std::string line;
    values.resize(size, std::vector<int>(sample_size));
    int sample_idx = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        for (int point_idx = 0; point_idx < sample_size; ++point_idx) {
            iss >> values[sample_idx][point_idx];
        }
        sample_idx++;
    }
    return true;
}

bool DataSubset::read_labels(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return false;
    }

    std::string line;
    labels.resize(size);
    int sample_idx = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        iss >> labels[sample_idx];
        sample_idx++;
    }
    return true;
}

/**
 * @brief Class representing the entire dataset, including training and test subsets.
 */
class Dataset {
public:
    int test_size; /**< The number of samples in the test set. */
    int train_size; /**< The number of samples in the train set. */
    int sample_size; /**< The number of points in each sample. */
    DataSubset train; /**< The training subset of the dataset. */
    DataSubset test; /**< The test subset of the dataset. */

    /**
     * @brief Reads the dataset parameters from a file.
     * 
     * @param filename The name of the file to read from.
     * @return True if the file was read successfully, false otherwise.
     */
    bool read_parameters(const std::string &filename);

    /**
     * @brief Loads the dataset from files.
     * 
     * @return 0 if the dataset was loaded successfully, 1 otherwise.
     */
    int load_dataset();

    /**
     * @brief Calculates the checksum of the dataset.
     * 
     * @return The calculated checksum.
     */
    int get_checksum();

    /**
     * @brief Returns the training set as a pair of values and labels.
     * 
     * @return A pair of vectors, where the first vector contains the values and the second vector contains the labels.
     */
    std::pair<std::vector<std::vector<int>>, std::vector<int>> get_trainset();

    /**
     * @brief Returns the test set as a pair of values and labels.
     * 
     * @return A pair of vectors, where the first vector contains the values and the second vector contains the labels.
     */
    std::pair<std::vector<std::vector<int>>, std::vector<int>> get_testset();
};

bool Dataset::read_parameters(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return false;
    }

    std::string line;
    std::getline(file, line);
    test_size = std::stoi(line);
    std::getline(file, line);
    train_size = std::stoi(line);
    std::getline(file, line);
    sample_size = std::stoi(line);
    train.size = train_size;
    train.sample_size = sample_size;
    test.size = test_size;
    test.sample_size = sample_size;
    file.close();
    return true;
}

int Dataset::load_dataset() {
    std::string base_path = "./dataset/" + std::string(DATASET) + "/";

    if (!read_parameters(base_path + "parameters")) {
        return 1;
    }

    if (!train.read_values(base_path + "train.val")) {
        return 1;
    }

    if (!train.read_labels(base_path + "train.label")) {
        return 1;
    }

    if (!test.read_values(base_path + "test.val")) {
        return 1;
    }

    if (!test.read_labels(base_path + "test.label")) {
        return 1;
    }

    return 0;
}

int Dataset::get_checksum() {
    int N = (1L << 20);
    int acc = 0;

    for (const auto &sample : train.values) {
        for (const auto &point : sample) {
            acc = (point + acc) % N;
        }
    }

    for (const auto &sample : test.values) {
        for (const auto &point : sample) {
            acc = (point + acc) % N;
        }
    }

    return acc;
}

std::pair<std::vector<std::vector<int>>, std::vector<int>> Dataset::get_trainset() {
    return std::make_pair(train.values, train.labels);
}

std::pair<std::vector<std::vector<int>>, std::vector<int>> Dataset::get_testset() {
    return std::make_pair(test.values, test.labels);
}

#endif // DATASET_H
