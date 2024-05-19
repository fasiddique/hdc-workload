#include "utils.h"
#include <random>
#include <iostream>

/**
 * @brief Binarizes the input vector.
 * 
 * @param x Input vector of integers.
 * @return A vector with values converted to 1 or -1 based on their sign.
 */
std::vector<int> binarize(const std::vector<int>& x) {
    std::vector<int> result;
    result.reserve(x.size());
    for (int val : x) {
        result.push_back(val > 0 ? 1 : -1);
    }
    return result;
}


/**
 * @brief Generates a random integer vector.
 * 
 * @param size The size of the vector to generate.
 * @param min The minimum value in the range of random integers.
 * @param max The maximum value in the range of random integers.
 * @return A vector of random integers.
 */
std::vector<int> generate_random_vector(int size, int min, int max) {
    std::vector<int> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min, max);
    for (int &val : vec) {
        val = dis(gen);
    }
    return vec;
}

/**
 * @brief Prints a 2D vector to the standard output.
 * 
 * @param vec The 2D vector to print.
 */
void print_2d_vector(const std::vector<std::vector<int>>& vec) {
    for (const auto& row : vec) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}
