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
    auto output = secure_index(arr_c, index_c, info);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    ptxt answer = decrypt(info, output);
    
    std::cout << "[ ";
    std::cout << "[ " << answer[0] << " ] (" << duration.count() << "ms)\n";
    
    return 0;

}