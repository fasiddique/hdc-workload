#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <stdint.h>
// Binarize function
std::vector<int> binarize(const std::vector<int>& x);

// Generate random integer vector
std::vector<int> generate_random_vector(int size, int min, int max);

// Print 2D vector
void print_2d_vector(const std::vector<std::vector<int>>& vec);

void gemv(uint64_t row, uint64_t col, const std::vector<int> &srcVector, const std::vector<std::vector<int>> &srcMatrix, std::vector<int> &dst); 

void gemm(uint64_t row, uint64_t colA, uint64_t colB, const std::vector<std::vector<int>> &srcMatrixA, const std::vector<std::vector<int>> &srcMatrixB, std::vector<std::vector<int>> &dstMatrix, bool shouldVerify);

#endif // UTILS_H
