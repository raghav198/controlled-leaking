#include <chrono>
#include <helib/binaryCompare.h>
#include <helib/intraSlot.h>

#include "common.hpp"

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

    std::cout << bitwidth << "\n";

    std::vector<long> arr_p = {14, 15, 9, 13, 6, 16, 19, 10};
    long thresh_p = 11;

    std::vector<ctxt> arr_c;
    ctxt thresh_c;

    for (auto val : arr_p)
        arr_c.push_back(encrypt(info, ptxt{val}));

    thresh_c = encrypt(info, ptxt{thresh_p});

    ctxt_bit scratch = encrypt_vector(info, std::vector<long>());
    ctxt_bit lt(scratch);
    ctxt_bit gt(scratch);

    ctxt zero = encrypt(info, ptxt{0});

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<ctxt> output;
    for (auto val : arr_c) {
        std::cout << "Filter step\n";
        helib::compareTwoNumbers(gt, lt, helib::CtPtrs_vectorCt(val), helib::CtPtrs_vectorCt(thresh_c));
        output.push_back(mux(lt, gt, zero, val));
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::vector<ptxt> answer;
    for (auto val : output)
        answer.push_back(decrypt(info, val));
    
    std::cout << "[ ";
    for (auto val : answer) std::cout << val[0] << " ";
    std::cout << "] (" << duration.count() << "ms)\n";
    
    return 0;

}