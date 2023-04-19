#pragma once

#include "coyote-runtime.hpp"

#include <vector>
#include <string>

struct CoyoteKernel {
    EncInfo& info;
    int width;

    CoyoteKernel(EncInfo& info, int width) : info(info), width(width) {}

    std::vector<ctxt> input_wires, output_wires;
    std::unordered_map<std::string, helib::Ptxt<helib::BGV>> masks;

    virtual void Prepare(std::unordered_map<std::string, int>) = 0; // populate input_wires from values assigned to variables and populate masks
    virtual void Compute() = 0; // populate output_wires from input_wires

    void add_masks(std::initializer_list<std::string> mask_list) {
        for (auto mask : mask_list) {
            std::vector<long> mask_data;
            for (auto c : mask) {
                mask_data.push_back(c - '0');
            }
            masks[mask] = zzx_vec(info.context, mask_data);
        }
    }

    virtual ~CoyoteKernel() {};

    std::vector<int> get_output(int index) {
        ptxt_vec result = decrypt(info, output_wires[index]);
        return std::vector<int>(result.begin(), result.begin() + width);
    }
};

