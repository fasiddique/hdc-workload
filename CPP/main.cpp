#include <iostream> 
#include <fstream>
#include <string>
#include <sstream>
#include "dataset.h"

int test_dataset() {
    pdataset_t pdataset = (pdataset_t)malloc(sizeof(dataset_t));
    if(load_dataset(pdataset)) {

        std::cout << "Error loading the dataset ..." << std::endl;
        return 1;
    }
    std::cout << "val: " << pdataset->train_val[0][0] << std::endl;
    std::cout << "Checksum: " << get_checksum(pdataset) << std::endl;
    std::cout << "test_size: " << pdataset->test_size << std::endl;
    std::cout << "train_size: " << pdataset->train_size << std::endl;
    std::cout << "sample_size: " << pdataset->sample_size << std::endl;
    return 0;
}


int main() {
    return test_dataset(); 
}