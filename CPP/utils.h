#ifndef UTILS_H
#define UTILS_H

#include <vector>

// Binarize function
std::vector<int> binarize(const std::vector<int>& x);

// Generate random integer vector
std::vector<int> generate_random_vector(int size, int min, int max);

// Print 2D vector
void print_2d_vector(const std::vector<std::vector<int>>& vec);

#endif // UTILS_H
