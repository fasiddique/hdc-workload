#include "utils.h"
#include <random>
#include <iostream>
#include <iostream>
#include <vector>
#include <stdint.h>
#include "libpimsim.h"

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



void gemv(uint64_t row, uint64_t col, const std::vector<int> &srcVector, const std::vector<std::vector<int>> &srcMatrix, std::vector<int> &dst)
{
  unsigned bitsPerElement = sizeof(int) * 8;
  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, row, bitsPerElement, PIM_INT32);
  if (srcObj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  PimObjId srcObj2 = pimAllocAssociated(bitsPerElement, srcObj1, PIM_INT32);
  if (srcObj2 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId dstObj = pimAllocAssociated(bitsPerElement, srcObj1, PIM_INT32);
  if (dstObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimStatus status = pimBroadcast(dstObj, 0);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  for (int i = 0; i < col; ++i)
  {
    status = pimCopyHostToDevice((void *)srcMatrix[i].data(), srcObj1);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimBroadcast(srcObj2, srcVector[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimMul(srcObj1, srcObj2, srcObj2);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimAdd(srcObj2, dstObj, dstObj);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
  }

  dst.reserve(row);
  status = pimCopyDeviceToHost(dstObj, (void *)dst.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
  }
  pimFree(srcObj1);
  pimFree(srcObj2);
  pimFree(dstObj);
}

void gemm(uint64_t row, uint64_t colA, uint64_t colB, const std::vector<std::vector<int>> &srcMatrixA, const std::vector<std::vector<int>> &srcMatrixB, std::vector<std::vector<int>> &dstMatrix, bool shouldVerify)
{
  //the result matrix is saved in transformed way
  dstMatrix.resize(colB, std::vector<int>(row, 0));
  for (int i = 0; i < colA; ++i)
  {
    gemv(row, colA, srcMatrixB[i], srcMatrixA, dstMatrix[i]);
  }
}

