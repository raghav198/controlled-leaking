#include <chrono>
#include <helib/binaryCompare.h>
#include <helib/intraSlot.h>

#include "common.hpp"

ctxt assoc_array_lookup_helper(std::vector<ctxt> keys, std::vector<ctxt> vals, ctxt key, int index, EncInfo & info)
{
    if (index == keys.size()) {
        return encrypt(info, ptxt{0});
    }
    ctxt_bit eq = secure_eq(key, keys[index], info);
    ctxt_bit neq = eq;
    neq.addConstant(NTL::ZZX(1));
    return mux(eq, neq, vals[index], assoc_array_lookup_helper(keys, vals, key, index + 1, info));
}

ctxt assoc_array_lookup(std::vector<ctxt> keys, std::vector<ctxt> vals, ctxt key, EncInfo & info)
{
    return assoc_array_lookup_helper(keys, vals, key, 0, info);
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

    std::vector<long> keys_p = {2, 14, 4, 9, 3, 19, 15, 1};
    std::vector<long> vals_p = {2, 7, 12, 8, 13, 15, 5, 15};
    long lookup_p = 9;

    std::vector<ctxt> keys_c, vals_c;
    ctxt lookup_c;

    for (auto val : keys_p)
        keys_c.push_back(encrypt(info, ptxt{val}));

    for (auto val : vals_p)
        vals_c.push_back(encrypt(info, ptxt{val}));

    lookup_c = encrypt(info, ptxt{lookup_p});

    ctxt_bit scratch = encrypt_vector(info, std::vector<long>());
    ctxt_bit lt(scratch);
    ctxt_bit gt(scratch);

    ctxt zero = encrypt(info, ptxt{0});

    auto start = std::chrono::high_resolution_clock::now();
    auto output = assoc_array_lookup(keys_c, vals_c, lookup_c, info);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    ptxt answer = decrypt(info, output);
    
    std::cout << "[ " << answer[0] << " ] (" << duration.count() << "ms)\n";
    
    return 0;

}