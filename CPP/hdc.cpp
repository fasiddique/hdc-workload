#include <cassert>
#include <cstddef> 
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

#include "hdc.h"
#include "utils.h"




HDC::HDC(int n_class, int n_lv, int n_id, int n_dim, bool binary)
    : n_class(n_class), n_lv(n_lv), n_id(n_id), n_dim(n_dim), binary(binary) {
    // Initialize hv_lv and hv_id with random values
    hv_lv = generate_hvs(n_lv, n_dim);
    hv_id = generate_hvs(n_id, n_dim);
    class_hvs = std::vector<std::vector<int>>(n_class, std::vector<int>(n_dim, 0));
}

std::vector<std::vector<int>> HDC::encode(const std::vector<std::vector<int>>& inp) {
    int n_batch = inp.size();
    std::vector<std::vector<int>> inp_enc(n_batch, std::vector<int>(n_dim, 0));

    for (int i = 0; i < n_batch; ++i) {
        std::vector<int> tmp(n_dim, 0);
        for (int j = 0; j < n_id; ++j) {
            for (int d = 0; d < n_dim; ++d) {
                tmp[d] += hv_id[j][d] * hv_lv[inp[i][j]][d];
            }
        }
        inp_enc[i] = tmp;
    }

    if (binary) {
        for (auto& vec : inp_enc) {
            vec = binarize(vec);
        }
    }

    return inp_enc;
}

// std::vector<std::vector<int>> HDC::generate_hvs(int n, int dim) {
//     std::vector<std::vector<int>> hvs(n);
//     for (int i = 0; i < n; ++i) {
//         hvs[i] = generate_random_vector(dim, -1, 1);
//     }
//     return hvs;
// }

std::vector<std::vector<int>> HDC::generate_hvs(int n, int dim) {
    int fixed_value = 2;
    std::vector<std::vector<int>> hvs(n, std::vector<int>(dim, fixed_value)); // Initialize with fixed value
    return hvs;
}



void HDC::train_init(const std::vector<std::vector<int>>& inp_enc, const std::vector<int>& target) {
    assert(inp_enc.size() == target.size());

    for (int i = 0; i < n_class; ++i) {
        std::vector<int> sum(n_dim, 0);
        for (size_t j = 0; j < target.size(); ++j) {
            if (target[j] == i) {
                for (int d = 0; d < n_dim; ++d) {
                    sum[d] += inp_enc[j][d];
                }
            }
        }
        class_hvs[i] = binary ? binarize(sum) : sum;
    }
}

// Implementation of the getter function
const std::vector<std::vector<int>>& HDC::get_class_hvs() const {
    return class_hvs;
}



double HDC::test(const std::vector<std::vector<int>>& inp_enc, const std::vector<int>& target) {
    assert(inp_enc.size() == target.size());

    std::vector<std::vector<double>> dist(inp_enc.size(), std::vector<double>(n_class, 0.0));

    // Distance matching
    for (size_t i = 0; i < inp_enc.size(); ++i) {
        for (int j = 0; j < n_class; ++j) {
            double dot_product = 0.0;
            for (int d = 0; d < n_dim; ++d) {
                if (binary) {
                    dot_product += inp_enc[i][d] * binarize(class_hvs[j])[d];
                } else {
                    dot_product += inp_enc[i][d] * class_hvs[j][d];
                }
            }
            if (!binary) {
                double norm = 0.0;
                for (int d = 0; d < n_dim; ++d) {
                    norm += class_hvs[j][d] * class_hvs[j][d];
                }
                norm = std::sqrt(norm);
                dist[i][j] = dot_product / norm;
            } else {
                dist[i][j] = dot_product;
            }
        }
    }

    int correct = 0;
    for (size_t i = 0; i < dist.size(); ++i) {
        int predicted = std::distance(dist[i].begin(), std::max_element(dist[i].begin(), dist[i].end()));
        if (predicted == target[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / target.size();
}



void HDC::train(const std::vector<std::vector<int>>& inp_enc, const std::vector<int>& target) {
    assert(inp_enc.size() == target.size());

    size_t n_samples = inp_enc.size();

    for (size_t j = 0; j < n_samples; ++j) {
        int pred = 0;
        if (binary) {
            std::vector<int> bin_class_hvs_flat;
            for (const auto& hv : class_hvs) {
                std::vector<int> bin_hv = binarize(hv);
                bin_class_hvs_flat.insert(bin_class_hvs_flat.end(), bin_hv.begin(), bin_hv.end());
            }
            std::vector<int> inp_enc_binarized = binarize(inp_enc[j]);
            std::vector<int> dist(n_class, 0);
            for (int i = 0; i < n_class; ++i) {
                int dot_product = 0;
                for (int d = 0; d < n_dim; ++d) {
                    dot_product += inp_enc_binarized[d] * bin_class_hvs_flat[i * n_dim + d];
                }
                dist[i] = dot_product;
            }
            pred = std::distance(dist.begin(), std::max_element(dist.begin(), dist.end()));
        } else {
            std::vector<double> dist(n_class, 0.0);
            for (int i = 0; i < n_class; ++i) {
                double dot_product = 0.0;
                for (int d = 0; d < n_dim; ++d) {
                    dot_product += inp_enc[j][d] * class_hvs[i][d];
                }
                double norm = 0.0;
                for (int d = 0; d < n_dim; ++d) {
                    norm += class_hvs[i][d] * class_hvs[i][d];
                }
                norm = std::sqrt(norm);
                dist[i] = dot_product / norm;
            }
            pred = std::distance(dist.begin(), std::max_element(dist.begin(), dist.end()));
        }

        if (pred != target[j]) {
            for (int d = 0; d < n_dim; ++d) {
                class_hvs[target[j]][d] += inp_enc[j][d];
                class_hvs[pred][d] -= inp_enc[j][d];
            }
        }
    }
}
