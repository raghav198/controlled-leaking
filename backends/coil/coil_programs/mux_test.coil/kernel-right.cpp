
#include "kernel.hpp"
COILRightKernel::COILRightKernel(EncInfo& info) : CoyoteKernel(info, 3) {}

void COILRightKernel::Prepare(std::unordered_map<std::string, int> inputs) {
    ptxt t0{inputs["input#3"], inputs["input#3"], inputs["input#3"]};
        input_wires.push_back(encrypt(info, t0));
        add_masks({});
}

void COILRightKernel::Compute() {
    output_wires.push_back(input_wires[0]);
}
    