
#include "kernel.hpp"
COILLabels::COILLabels(EncInfo& info) : CoyoteKernel(info, 17) {}

void COILLabels::Prepare(std::unordered_map<std::string, int> inputs) {
    
    int positive_pads = 0;
    int negative_pads = 0;
    
    ptxt t0{inputs["input#0"], inputs["input#1"], inputs["input#2"], inputs["input#3"], inputs["input#4"], inputs["input#5"], inputs["input#6"], inputs["input#7"], inputs["input#8"], inputs["input#9"], inputs["input#10"], inputs["input#11"], inputs["input#12"], inputs["input#13"], inputs["input#14"], inputs["input#15"], 0};
        input_wires.push_back(encrypt(info, t0, positive_pads, negative_pads));
        add_masks({});
}

void COILLabels::Compute() {
    output_wires.push_back(input_wires[0]);
}
    