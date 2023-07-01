#include <array>
#include <chrono>
#include <helib/FHE.h>
#include <helib/binaryArith.h>
#include <helib/binaryCompare.h>
#include <helib/intraSlot.h>
#include <iostream>
#include <vector>
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
    
    EncInfo info(8191, 2, 1, 500, 2);

    std::cout << info.nslots << "slots\n";
 
#ifdef VECTREE_THREADED
    NTL::SetNumThreads(32);
#endif

    input_handler inp;
    // inp.add_arr({2, 10, 11, 6, 8, 9, 4}); // the list
    // inp.add_num(7); // threshold below which stuff gets zeroed out

    inp.add_arr({2, 4, 3, 1, 5, 6, 8, 6, 7, 9});
    inp.add_num(3);
    inp.add_num(4);

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

    auto computed = std::chrono::high_resolution_clock::now();

    if (left_kernel.output_wires.size() == 0) {
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
        helib::CtPtrs_vectorCt(left_kernel.output_wires[0]), 
        helib::CtPtrs_vectorCt(right_kernel.output_wires[0]), true);

    ctxt_bit eq = lt;
    eq += op;
    eq += NTL::ZZ(1);

    auto lt_mask_ptxt = make_mask(info, maurice.lt_mask());
    auto eq_mask_ptxt = make_mask(info, maurice.eq_mask());

    lt *= lt_mask_ptxt;
    eq *= eq_mask_ptxt;

    ctxt_bit decisions(lt);
    decisions += eq;

#ifdef DEBUG
    for (auto i : left_kernel.get_output(0))
        std::cout << i << " ";
    std::cout << "\n";

    std::cout << "vs\n";

    for (auto i : right_kernel.get_output(0))
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