
#include "kernel.hpp"
COILLeftKernel::COILLeftKernel(EncInfo& info) : CoyoteKernel(info, 15) {}

void COILLeftKernel::Prepare(std::unordered_map<std::string, int> inputs) {
    
    int positive_pads = 0;
    int negative_pads = 1;
    
    ptxt t0{inputs["input#16"], inputs["input#16"], inputs["input#16"], inputs["input#16"], inputs["input#16"], inputs["input#16"], inputs["input#16"], inputs["input#16"], inputs["input#16"], inputs["input#16"], inputs["input#16"], inputs["input#16"], inputs["input#16"], inputs["input#16"], inputs["input#16"]};
        input_wires.push_back(encrypt(info, t0, positive_pads, negative_pads));
        add_masks({});
}

void COILLeftKernel::Compute() {
    output_wires.push_back(input_wires[0]);
}
    