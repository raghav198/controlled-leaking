
#include "kernel.hpp"
COILLabels::COILLabels(EncInfo& info) : CoyoteKernel(info, 11) {}

void COILLabels::Prepare(std::unordered_map<std::string, int> inputs) {
    
    int positive_pads = 0;
    int negative_pads = 0;
    
    ptxt t0{1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1};
        input_wires.push_back(encrypt(info, t0, positive_pads, negative_pads));
        add_masks({});
}

void COILLabels::Compute() {
    output_wires.push_back(input_wires[0]);
}
    