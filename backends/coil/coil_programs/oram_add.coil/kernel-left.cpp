
#include "kernel.hpp"
COILLeftKernel::COILLeftKernel(EncInfo& info) : CoyoteKernel(info, 66) {}

void COILLeftKernel::Prepare(std::unordered_map<std::string, int> inputs) {
    ptxt t0{9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 7, 6, 5, 4, 3, 2, 0, 9, 8, 7, 6, 5, 4, 3, 0, 9, 8, 7, 6, 5, 4, 0, 9, 8, 7, 6, 5, 0, 9, 8, 7, 6, 0, 9, 8, 7, 0, 9, 8, 0, 9, 0, 0, 0, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
        input_wires.push_back(encrypt(info, t0));
        add_masks({});
}

void COILLeftKernel::Compute() {
    output_wires.push_back(input_wires[0]);
}
    