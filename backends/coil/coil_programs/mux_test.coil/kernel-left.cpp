
#include "kernel.hpp"
COILLeftKernel::COILLeftKernel(EncInfo& info) : CoyoteKernel(info, 5) {}

void COILLeftKernel::Prepare(std::unordered_map<std::string, int> inputs) {
    ptxt t0{4, 3, 2, 1, 0};
        input_wires.push_back(encrypt(info, t0));
        add_masks({});
}

void COILLeftKernel::Compute() {
    output_wires.push_back(input_wires[0]);
}
    