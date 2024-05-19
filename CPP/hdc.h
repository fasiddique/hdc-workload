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
    
    /**
    * @brief Initializes the class hypervectors based on encoded inputs and target labels.
    * @param inp_enc Encoded input data.
    * @param target Target labels.
    */
    void train_init(const std::vector<std::vector<int>>& inp_enc, const std::vector<int>& target);

    /**
     * @brief Getter for the class hypervectors.
     * @return Class hypervectors.
     */
    const std::vector<std::vector<int>>& get_class_hvs() const;


    /**
    * @brief Computes the accuracy of the model on the test data.
    *
    * This function calculates the dot product between the input encodings and the class hypervectors.
    * If the model is not binary, it normalizes the distances. Finally, it computes the accuracy by
    * comparing the predicted labels to the target labels.
    *
    * @param inp_enc The encoded input data to be tested.
    * @param target The target labels for the input data.
    * @return The accuracy of the model on the test data.
    */
    double test(const std::vector<std::vector<int>>& inp_enc, const std::vector<int>& target);


    /**
    * @brief Trains the HDC model using the input encodings and target labels.
    *
    * This function updates the class hypervectors based on the input encodings and
    * target labels. If the predicted label does not match the target label, the
    * function updates the class hypervectors accordingly.
    *
    * @param inp_enc The encoded input data.
    * @param target The target labels for the input data.
    */
    void train(const std::vector<std::vector<int>>& inp_enc, const std::vector<int>& target);

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
