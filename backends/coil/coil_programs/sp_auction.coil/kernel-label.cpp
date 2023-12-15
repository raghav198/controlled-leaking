
#include "kernel.hpp"
COILLabels::COILLabels(EncInfo& info) : CoyoteKernel(info, 128) {}

void COILLabels::Prepare(std::unordered_map<std::string, int> inputs) {
    
    int positive_pads = 0;
    int negative_pads = 0;
    
    ptxt t0{7, 6, 7, 5, 7, 6, 7, 4, 7, 6, 7, 5, 7, 6, 7, 3, 7, 6, 7, 5, 7, 6, 7, 4, 7, 6, 7, 5, 7, 6, 7, 2, 7, 6, 7, 5, 7, 6, 7, 4, 7, 6, 7, 5, 7, 6, 7, 3, 7, 6, 7, 5, 7, 6, 7, 4, 7, 6, 7, 5, 7, 6, 7, 1, 7, 6, 7, 5, 7, 6, 7, 4, 7, 6, 7, 5, 7, 6, 7, 3, 7, 6, 7, 5, 7, 6, 7, 4, 7, 6, 7, 5, 7, 6, 7, 2, 7, 6, 7, 5, 7, 6, 7, 4, 7, 6, 7, 5, 7, 6, 7, 3, 7, 6, 7, 5, 7, 6, 7, 4, 7, 6, 7, 5, 7, 6, 7, 0};
        input_wires.push_back(encrypt(info, t0, positive_pads, negative_pads));
        ptxt t1{inputs["input#6"], inputs["input#5"], inputs["input#5"], inputs["input#4"], inputs["input#6"], inputs["input#4"], inputs["input#4"], inputs["input#3"], inputs["input#6"], inputs["input#5"], inputs["input#5"], inputs["input#3"], inputs["input#6"], inputs["input#3"], inputs["input#3"], inputs["input#2"], inputs["input#6"], inputs["input#5"], inputs["input#5"], inputs["input#4"], inputs["input#6"], inputs["input#4"], inputs["input#4"], inputs["input#2"], inputs["input#6"], inputs["input#5"], inputs["input#5"], inputs["input#2"], inputs["input#6"], inputs["input#2"], inputs["input#2"], inputs["input#1"], inputs["input#6"], inputs["input#5"], inputs["input#5"], inputs["input#4"], inputs["input#6"], inputs["input#4"], inputs["input#4"], inputs["input#3"], inputs["input#6"], inputs["input#5"], inputs["input#5"], inputs["input#3"], inputs["input#6"], inputs["input#3"], inputs["input#3"], inputs["input#1"], inputs["input#6"], inputs["input#5"], inputs["input#5"], inputs["input#4"], inputs["input#6"], inputs["input#4"], inputs["input#4"], inputs["input#1"], inputs["input#6"], inputs["input#5"], inputs["input#5"], inputs["input#1"], inputs["input#6"], inputs["input#1"], inputs["input#1"], inputs["input#0"], inputs["input#6"], inputs["input#5"], inputs["input#5"], inputs["input#4"], inputs["input#6"], inputs["input#4"], inputs["input#4"], inputs["input#3"], inputs["input#6"], inputs["input#5"], inputs["input#5"], inputs["input#3"], inputs["input#6"], inputs["input#3"], inputs["input#3"], inputs["input#2"], inputs["input#6"], inputs["input#5"], inputs["input#5"], inputs["input#4"], inputs["input#6"], inputs["input#4"], inputs["input#4"], inputs["input#2"], inputs["input#6"], inputs["input#5"], inputs["input#5"], inputs["input#2"], inputs["input#6"], inputs["input#2"], inputs["input#2"], inputs["input#0"], inputs["input#6"], inputs["input#5"], inputs["input#5"], inputs["input#4"], inputs["input#6"], inputs["input#4"], inputs["input#4"], inputs["input#3"], inputs["input#6"], inputs["input#5"], inputs["input#5"], inputs["input#3"], inputs["input#6"], inputs["input#3"], inputs["input#3"], inputs["input#0"], inputs["input#6"], inputs["input#5"], inputs["input#5"], inputs["input#4"], inputs["input#6"], inputs["input#4"], inputs["input#4"], inputs["input#0"], inputs["input#6"], inputs["input#5"], inputs["input#5"], inputs["input#0"], inputs["input#6"], inputs["input#0"], inputs["input#0"], inputs["input#0"]};
        input_wires.push_back(encrypt(info, t1, positive_pads, negative_pads));
        add_masks({});
}

void COILLabels::Compute() {
    output_wires.push_back(input_wires[0]);
        output_wires.push_back(input_wires[1]);
}
    