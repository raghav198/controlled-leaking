
#include "kernel.hpp"
COILRightKernel::COILRightKernel(EncInfo& info) : CoyoteKernel(info, 66) {}

void COILRightKernel::Prepare(std::unordered_map<std::string, int> inputs) {
    ptxt t0{inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        input_wires.push_back(encrypt(info, t0));
        ptxt t1{inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        input_wires.push_back(encrypt(info, t1));
        ptxt t2{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"], inputs["input#10"]};
        input_wires.push_back(encrypt(info, t2));
        ptxt t3{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"], inputs["input#11"]};
        input_wires.push_back(encrypt(info, t3));
        add_masks({"000000000000000000000000000000000000000000000000000000001111111111", "111111111111111111111111111111111111111111111111111111110000000000"});
}

void COILRightKernel::Compute() {
    auto v0 = mul(input_wires[0], input_wires[1]);
        auto v1 = add(input_wires[2], input_wires[3]);
        auto v2 = blend({{v0, masks["111111111111111111111111111111111111111111111111111111110000000000"]}, {v1, masks["000000000000000000000000000000000000000000000000000000001111111111"]}});
        output_wires.push_back(v2);
}
    