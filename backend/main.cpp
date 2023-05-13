#include <array>
#include <helib/FHE.h>
#include <helib/binaryArith.h>
#include <helib/binaryCompare.h>
#include <iostream>
#include <vector>
#include <copse/vectrees.hpp>
#include <copse/sally-server.hpp>

#include "gcd.coil/gcd.hpp"

int extract(EncInfo& info, ctxt wire) {
    auto answer = decrypt(info, wire);
    for (auto i : answer) {
        if (i != 0) {
            return i;
        }
    }
    return 0;
}

ctxt_bit my_reduce(std::vector<ctxt_bit> vals, std::function<ctxt_bit(ctxt_bit, ctxt_bit)> func, int start, int end)
{

    if (end - start == 0 || vals.size() == 0)
        throw new std::runtime_error("Cannot reduce size-0 vector");
    if (end - start == 1)
        return vals[start];

    int mid = start + (end - start) / 2;

    auto lhs = my_reduce(vals, func, start, mid);
    auto rhs = my_reduce(vals, func, mid, end);

    auto answer = func(lhs, rhs);

    return func(lhs, rhs);
}

template<typename T>
std::vector<T> slice(std::vector<T> input, int start, int len) {
    return std::vector<T>(input.begin() + start, input.begin() + start + len);
}

int main(int argc, char * argv[])
{

    EncInfo info(4095, 2, 1, 500, 2);

    std::unordered_map<std::string, int> inputs = {{"x0", 4}, {"x1", 2}, {"x2", 5}, {"a0", 12}, {"b0", 18}};
    ctxt_bit scratch = encrypt_vector(info, std::vector<long>());

    COILMaurice maurice(info);
    auto model = maurice.GenerateModel();

    COILLeftKernel left_kernel(info);
    left_kernel.Prepare(inputs);
    left_kernel.Compute();

    COILRightKernel right_kernel(info);
    right_kernel.Prepare(inputs);
    right_kernel.Compute();

    ctxt_bit decisions(scratch);
    ctxt_bit op(scratch);
    helib::compareTwoNumbers(op, decisions, 
        helib::CtPtrs_vectorCt(left_kernel.output_wires[0]), 
        helib::CtPtrs_vectorCt(right_kernel.output_wires[0]), true);

    ctxt_bit eq = decisions;
    eq += op;
    eq += NTL::ZZ(1);

#ifdef DEBUG
    for (auto i : left_kernel.get_output(0))
        std::cout << i << " ";
    std::cout << "\n";

    std::cout << "<=>\n";

    for (auto i : right_kernel.get_output(0))
        std::cout << i << " ";
    std::cout << "\n";

    std::cout << "---\n";

    for (auto i : slice(decrypt_vector(info, decisions), 0, left_kernel.width))
        std::cout << i << " ";
    std::cout << "\n";

    for (auto i : slice(decrypt_vector(info, eq), 0, left_kernel.width))
        std::cout << i << " ";
    std::cout << "\n";
#endif

    int num_levels = model->level_b2s.size();
    std::vector<ctxt_bit> masks(num_levels, decisions);
    auto branch_rots = generate_rotations(decisions, model->level_b2s[0].size(), info.context.getEA());

    for (int i = 0; i < num_levels; i++) {
        auto b2s = model->level_b2s[i];
        auto mask = model->level_mask[i];

#ifdef DEBUG
        std::cout << "Current level matrix:\n";
        for (auto diag : b2s) {
            for (auto bit : slice(decrypt_vector(info, diag), 0, left_kernel.width + 1)) 
                std::cout << bit << " ";
            std::cout << "\n";
        }

        std::cout << "Branch rotations:\n";
        for (auto rot : branch_rots) {
            for (auto bit : slice(decrypt_vector(info, rot), 0, left_kernel.width)) 
                std::cout << bit << " ";
            std::cout << "\n";
        }
#endif
        auto slots = mat_mul(b2s, branch_rots);

#ifdef DEBUG
        std::cout << "matmul result: ";
        for (auto bit : slice(decrypt_vector(info, slots), 0, left_kernel.width + 1)) 
            std::cout << bit << " ";
        std::cout << "\n";
#endif
        slots += mask;
        masks[i] = slots;
    }

#ifdef DEBUG
    std::cout << "Resulting masks:\n";
    for (auto mask : masks) {
        std::cout << "* ";
        for (auto bit : slice(decrypt_vector(info, mask), 0, left_kernel.width + 1)) {
            std::cout << bit << " ";
        }
        std::cout << "\n";
    }
#endif

    auto inference = my_reduce(
        masks,
        [](auto lhs, auto rhs) {
            lhs.multiplyBy(rhs);
            return lhs;
        },
        0, masks.size());

#ifdef DEBUG
    std::cout << "inference:\n";
    for (auto bit : decrypt_vector(info, inference)) {
        std::cout << bit << " ";
    }
    std::cout << "\n";
#endif

    COILLabels label_kernel(info);
    label_kernel.Prepare(inputs);
    label_kernel.Compute();

#ifdef DEBUG
    std::cout << "Decrypting results...\n";
    
    std::cout << "Computed labels:\n";
    for (int i = 0; i < label_kernel.output_wires.size(); i++) {
        for (auto x : label_kernel.get_output(i)) {
            std::cout << x << " ";
        }
        std::cout << "\n";
    }
#endif

    std::vector<ctxt> answer_enc_array;
    for (auto output : label_kernel.output_wires) {
        ctxt answer_enc;
        for (auto wire : output) {
            auto cur_bit = wire;
            cur_bit *= inference;
            answer_enc.push_back(cur_bit);
        }
        answer_enc_array.push_back(answer_enc);
    }
    
    std::cout << "Answer: [ ";

    for (auto answer_enc : answer_enc_array) {
        std::cout << extract(info, answer_enc) << " ";
    }
    std::cout << "]\n";

    return 0;

}