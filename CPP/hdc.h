#ifndef HDC_H
#define HDC_H

#include <vector>


/**
 * @class HDC
 * @brief A class implementing Hyperdimensional Computing (HDC).
 */
class HDC {
public:
    /**
     * @brief Constructor for HDC class.
     * 
     * @param n_class Number of classes.
     * @param n_lv Number of level hypervectors.
     * @param n_id Number of identifier hypervectors.
     * @param n_dim Dimension of hypervectors.
     * @param binary Whether to use binary hypervectors.
     */
    HDC(int n_class, int n_lv, int n_id, int n_dim, bool binary);

    /**
     * @brief Encodes the input data into hyperdimensional vectors.
     * 
     * @param inp Input data to be encoded.
     * @return Encoded hyperdimensional vectors.
     */
    std::vector<std::vector<int>> encode(const std::vector<std::vector<int>>& inp);

private:
    int n_class; ///< Number of classes.
    int n_lv; ///< Number of level hypervectors.
    int n_id; ///< Number of identifier hypervectors.
    int n_dim; ///< Dimension of hypervectors.
    bool binary; ///< Whether to use binary hypervectors.

    std::vector<std::vector<int>> hv_lv; ///< Level hypervectors.
    std::vector<std::vector<int>> hv_id; ///< Identifier hypervectors.
    std::vector<std::vector<int>> class_hvs; ///< Class hypervectors.

    /**
     * @brief Generates a set of random hyperdimensional vectors.
     * 
     * @param n Number of hypervectors to generate.
     * @param dim Dimension of each hypervector.
     * @return Generated hyperdimensional vectors.
     */
    std::vector<std::vector<int>> generate_hvs(int n, int dim);
};

#endif // HDC_H
