#include <chrono>
#include <helib/binaryCompare.h>
#include <helib/intraSlot.h>

#include "common.hpp"

ctxt log_index_helper(std::vector<ctxt> arr, ctxt index, int low, int high, EncInfo & info)
{
   
    if (low == high - 1) return arr[low];

    int mid = (high + low) / 2;

    std::cout << "Indexing in range [" << low << ", " << high << "], mid = " << mid << "\n";

    ctxt low_answer = log_index_helper(arr, index, low, mid, info);
    ctxt high_answer = log_index_helper(arr, index, mid, high, info);

    // std::cout << "Low answer = " << decrypt(info, low_answer)[0] << "\n";
    // std::cout << "High answer = " << decrypt(info, high_answer)[0] << "\n";
    
    ctxt mid_enc = encrypt(info, ptxt{mid});
    ctxt_bit scratch = encrypt_vector(info, ptxt());
    ctxt_bit lt(scratch), gt(scratch);
    helib::compareTwoNumbers(gt, lt, helib::CtPtrs_vectorCt(index), helib::CtPtrs_vectorCt(mid_enc));
    gt = lt;
    gt.addConstant(NTL::ZZX(1));
    // std::cout << "Comparison: (gt, lt) = " << 

    return mux(lt, gt, low_answer, high_answer);
}

ctxt log_index(std::vector<ctxt> arr, ctxt index, EncInfo & info)
{
    return log_index_helper(arr, index, 0, arr.size(), info);
}

int main()
{
    auto contextBuilder = helib::ContextBuilder<helib::BGV>()
        .m(8191)
        .p(2)
        .r(1)
        .bits(500)
        .c(2);
    EncInfo info(contextBuilder);
    info.context.printout();

    std::vector<long> arr_p = {9, 2, 3, 12, 6, 8, 7, 1, 4, 5, 0, 10, 21, 16, 30, 13};
    long index_p = 11;

    std::vector<ctxt> arr_c;
    ctxt index_c;

    for (auto val : arr_p)
        arr_c.push_back(encrypt(info, ptxt{val}));

    index_c = encrypt(info, ptxt{index_p});

    ctxt_bit scratch = encrypt_vector(info, std::vector<long>());
    ctxt_bit lt(scratch);
    ctxt_bit gt(scratch);

    ctxt zero = encrypt(info, ptxt{0});

    auto start = std::chrono::high_resolution_clock::now();
    auto output = log_index(arr_c, index_c, info);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    ptxt answer = decrypt(info, output);
    
    std::cout << "[ " << answer[0] << " ] (" << duration.count() << "ms)\n";
    
    return 0;

}