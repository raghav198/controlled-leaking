#include <array>
#include <chrono>
#include <helib/FHE.h>
#include <helib/binaryArith.h>
#include <helib/binaryCompare.h>
#include <helib/intraSlot.h>
#include <iostream>
#include <vector>
#include <functional>
#include <copse/vectrees.hpp>
#include <copse/sally-server.hpp>

#include "kernel.hpp"
#include "inputs.hpp"

int extract(EncInfo& info, ctxt wire, int limit) {
    auto answer = decrypt(info, wire);
    int pos = 0;
    for (auto i : answer) {
        if (pos++ == limit) return 0;
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

#ifdef PTXT_MODEL
#define debug_vector decode_vector
#else
#define debug_vector decrypt_vector
#endif



int main(int argc, char * argv[])
{
    
    std::cout << "Setting up encryption..." << std::flush;
    // x.addConstant()
    EncInfo info(8191, 2, 1, 500, 2);
    // EncInfo info(55831, 2, 1, 1000, 2);

    std::cout << info.nslots << "slots\n";

#ifdef DEBUG
    std::cout << "Debug mode!\n";
#endif
 
#ifdef VECTREE_THREADED
    NTL::SetNumThreads(32);
#endif

    input_handler inp;
    
    // // linear_oram, log_oram
    // inp.add_arr({9, 2, 3, 12, 6, 8, 7, 1, 4, 5, 0, 10, 21, 16, 30, 13}); // array
    // inp.add_num(6); // index

    // inp.add_arr({1, 2, 3});
    // inp.add_num(4);

    // // filter
    // inp.add_arr({14, 15, 9, 13, 6, 16, 19, 10}); // array
    // inp.add_num(11); // threshold

    // merge
    inp.add_arr({7, 12, 17, 18, 20}); // array 1
    inp.add_arr({3, 5, 11, 14, 19}); // array 2

    // // sp_auction
    // inp.add_arr({13, 6, 11, 8, 3, 7, 17, 12}); // bids

    // // associative_array
    // inp.add_arr({2, 14, 4, 9, 3, 19, 15, 1}); // keys
    // inp.add_arr({2, 7, 12, 8, 13, 15, 5, 15}); // values
    // inp.add_num(4); // lookup

    // inp.add_arr({2, 5, 7, 8, 9, 13, 11, 6}); // array
    // inp.add_num(2); // i
    // inp.add_num(3); // j
    // inp.add_num(0); // k1
    // inp.add_num(1); // k2

    // GCD
    // inp.add_num(21);
    // inp.add_num(28);


    ctxt_bit scratch = encrypt_vector(info, std::vector<long>());

    std::cout << "Encrypting model...\n";
    COILMaurice maurice(info);
    CtxtModelDescription * model = maurice.GenerateModel();

    std::cout << "Encrypting inputs...\n";
    COILLeftKernel left_kernel(info);
    left_kernel.Prepare(inp.inputs);
    

    COILRightKernel right_kernel(info);
    right_kernel.Prepare(inp.inputs);
    

    COILLabels label_kernel(info);
    label_kernel.Prepare(inp.inputs);
    
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Computing inputs...\n";
    left_kernel.Compute();
    right_kernel.Compute();
    label_kernel.Compute();

    auto left_values = left_kernel.output_wires[0];
    auto right_values = right_kernel.output_wires[0];

    // if (left_kernel.input_wires[0].size() != left_kernel.width) {
        std::stringstream mask_string;
        for (int i = 0; i < left_kernel.width; i++) {
            mask_string << "1";
        }

        auto mask = make_mask(info, mask_string.str(), 0, 0);
        left_values = blend({{left_values, mask}});
        right_values = blend({{right_values, mask}});
        left_values = add(left_values, rotate(left_values, left_kernel.width));
        right_values = add(right_values, rotate(right_values, left_kernel.width));
    // }

    auto computed = std::chrono::high_resolution_clock::now();

    if (left_kernel.output_wires.size() == 0)
    {
        std::cout << "skipping branches\n";
        std::cout << "Answer: [ ";
        for (auto answer_enc : label_kernel.output_wires) {
            std::cout << extract(info, answer_enc, label_kernel.width) << " ";
        }
        std::cout << "]\n";

        return 0;
    }

    std::cout << "Computing branches...\n";
    ctxt_bit lt(scratch);
    ctxt_bit op(scratch);

    helib::compareTwoNumbers(op, lt, 
        helib::CtPtrs_vectorCt(left_values), 
        helib::CtPtrs_vectorCt(right_values), true);

    ctxt_bit eq = lt;
    eq += op;
    eq += NTL::ZZ(1);

    auto lt_mask_ptxt = make_mask(info, maurice.lt_mask(), 0, 1);
    auto eq_mask_ptxt = make_mask(info, maurice.eq_mask(), 0, 1);

    lt *= lt_mask_ptxt;
    eq *= eq_mask_ptxt;

    ctxt_bit decisions(lt);
    decisions += eq;

#ifdef DEBUG
    for (auto i : decrypt(info, left_values))
        std::cout << i << " ";
    std::cout << "\n";

    std::cout << "vs\n";

    for (auto i : decrypt(info, right_values))
        std::cout << i << " ";
    std::cout << "\n";

    std::cout << "---\n";

    for (auto i : slice(decrypt_vector(info, decisions), 0, left_kernel.width))
        std::cout << i << " ";
    std::cout << "<=>\n";
#endif

    int num_levels = model->level_b2s.size();
    std::vector<ctxt_bit> masks(num_levels, decisions);
    auto branch_rots = generate_rotations(decisions, model->level_b2s[0].size(), info.context.getEA());

#ifdef DEBUG
    std::cout << "Branch rotations:\n";
    for (auto rot : branch_rots) {
        for (auto bit : slice(decrypt_vector(info, rot), 0, left_kernel.width)) 
            std::cout << bit << " ";
        std::cout << "\n";
    }
#endif

    for (int i = 0; i < num_levels; i++) {
        std::cout << "Processing level " << i+1 << "/" << num_levels << "\n";
        auto b2s = model->level_b2s[i];
        auto mask = model->level_mask[i];

#ifdef DEBUG
        // std::cout << "Current level matrix:\n";
        // for (auto diag : b2s) {
        //     for (auto bit : slice(debug_vector(info, diag), 0, left_kernel.width + 1)) 
        //         std::cout << bit << " ";
        //     std::cout << "\n";
        // }
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

    std::cout << "Accumulating results...\n";
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
    
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Answer: [ ";

    for (auto answer_enc : answer_enc_array) {
        std::cout << extract(info, answer_enc, label_kernel.width) << " ";
    }
    std::cout << "] (" << duration.count() << "ms)\n";
    std::cout << "Coyote time: " << std::chrono::duration_cast<std::chrono::milliseconds>(computed - start).count() << "ms\n";

    return 0;

}