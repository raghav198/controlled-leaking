
#include "kernel.hpp"
COILLabels::COILLabels(EncInfo& info) : CoyoteKernel(info, 6) {}

void COILLabels::Prepare(std::unordered_map<std::string, int> inputs) {
    ptxt t0{inputs["input#0"], inputs["input#1"], inputs["input#2"], inputs["input#3"], inputs["input#4"], 0};
        input_wires.push_back(encrypt(info, t0));
        add_masks({});
}

void COILLabels::Compute() {
    output_wires.push_back(input_wires[0]);
}
    