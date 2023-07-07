#ifndef MUX_COMMON_HPP
#define MUX_COMMON_HPP

#include <copse/vectrees.hpp>
#include <coyote-runtime.hpp>

struct compute_data {
    std::vector<zzx_vec> plaintexts;
    std::vector<ctxt_bit> ciphertexts;
    std::unordered_map<std::string, zzx_vec> masks;
};

std::vector<std::vector<int>> lanes();
compute_data Prep(EncInfo & info, std::unordered_map<std::string, int> inputs);
std::vector<ctxt_bit> Compute(EncInfo & info, compute_data data);


#endif