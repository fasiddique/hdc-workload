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
