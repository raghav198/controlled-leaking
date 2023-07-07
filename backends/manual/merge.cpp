#include <chrono>
#include <helib/binaryCompare.h>
#include <helib/intraSlot.h>

#include "common.hpp"


int main()
{
    auto contextBuilder = helib::ContextBuilder<helib::BGV>()
        .m(55831)
        .p(2)
        .r(1)
        .bits(1500) // 580
        .c(2)
        .mvec({31, 1801})
        .gens({19812, 50593})
        .ords({30, 72})
        // .buildCache(true)
        // .buildModChain(true)
        // .thinboot()
        .bootstrappable(true);

    EncInfo info(contextBuilder);
    info.context.printout();


    ctxt_bit scratch = encrypt_vector(info, std::vector<long>());
    ctxt_bit lt(scratch);
    ctxt_bit gt(scratch);

    // plaintext array inputs
    std::vector<long> arr1_p = {2, 5, 6};//, 11, 14};
    std::vector<long> arr2_p = {3, 4, 7};//, 10, 12};
    std::vector<ctxt> arr1_c;
    std::vector<ctxt> arr2_c;

    std::cout << "Recrypt index: " << info.sk->genRecryptData() << "\n";

    std::vector<helib::zzX> unpackSlotEncoding;
    helib::buildUnpackSlotEncoding(unpackSlotEncoding, info.context.getEA());
    

    // encrypt values
    for (auto p : arr1_p) {
        arr1_c.push_back(encrypt(info, ptxt{p}));
    }
    for (auto p : arr2_p) {
        arr2_c.push_back(encrypt(info, ptxt{p}));
    }
    auto arr1_size = encrypt(info, ptxt{static_cast<long>(arr1_c.size())});
    auto arr2_size = encrypt(info, ptxt{static_cast<long>(arr2_c.size())});

    auto one = encrypt(info, ptxt{1});
    auto i1 = encrypt(info, ptxt{0});
    auto i2 = encrypt(info, ptxt{0});

    std::cout << "Initial noise bounds: " << arr1_c[0][0].getNoiseBound() << ", capacity = " << arr1_c[0][0].capacity() << "\n";

    std::vector<ctxt> out_c;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < arr1_p.size() + arr2_p.size(); i++) {
        // Look up the two valus we're comparing
        ctxt arr1_val = secure_index(arr1_c, i1, info, &unpackSlotEncoding);
        ctxt arr2_val = secure_index(arr2_c, i2, info, &unpackSlotEncoding);

        std::cout << "Indexed ctxts; noise = " << arr1_val[0].capacity() << "; " << arr2_val[0].capacity() << "\n";

        helib::compareTwoNumbers(gt, lt, 
                        helib::CtPtrs_vectorCt(arr1_val), 
                        helib::CtPtrs_vectorCt(arr2_val), true, &unpackSlotEncoding);

        std::cout << "Comparison noise = " << gt.capacity() << "; " << lt.capacity() << "\n";

        // which, if any, array has run out?
        ctxt_bit arr1_end = secure_eq(i1, arr1_size, info, &unpackSlotEncoding);
        ctxt_bit arr1_mid = arr1_end;
        arr1_mid.addConstant(NTL::ZZX(1));
        
        ctxt_bit arr2_end = secure_eq(i2, arr2_size, info, &unpackSlotEncoding);
        ctxt_bit arr2_mid = arr2_end;
        arr2_mid.addConstant(NTL::ZZX(1));

        // Grab the output of the comparison...
        ctxt output = mux(lt, gt, arr1_val, arr2_val);
        // ...unless one of the arrays has run out
        output = mux(arr1_mid, arr1_end, output, arr2_val);
        output = mux(arr2_mid, arr2_end, output, arr1_val);

        out_c.push_back(output);

        std::cout << "Output noise = " << output[0].capacity() << "\n";

        // Decide which index to increment
        i1 = mux(lt, gt, add(i1, one), i1);
        i2 = mux(lt, gt, i2, add(i2, one));

        std::cout << "Index noise = " << i1[0].capacity() << "; " << i2[0].capacity() << "\n";
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::vector<ptxt> out_p;
    for (auto c : out_c) {
        out_p.push_back(decrypt(info, c));
    }
    std::cout << "[ ";
    for (auto val : out_p) {
        std::cout << val[0] << " ";
    }
    std::cout << "] (" << duration.count() << "ms)\n";
    return 0;

}
