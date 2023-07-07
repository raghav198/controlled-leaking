#include "common.hpp"

ctxt get_ctxt(EncInfo & info, long val)
{
    std::vector<long> ptxt { val };
    ctxt encrypted;
    for (auto bit : decompose_bits(ptxt, 8)) {
        encrypted.push_back(encrypt_vector(info, bit));
    }
    return encrypted;
}

ctxt mux(ctxt_bit l_bit, ctxt_bit r_bit, ctxt left, ctxt right) {
    ctxt out;
    for (int i = 0; i < left.size(); i++) {
        left[i].multiplyBy(l_bit);
        out.push_back(left[i]);
        right[i].multiplyBy(r_bit);
        out[i] += right[i];
    }
    return out;
}

template<typename T>
void print_vector(std::vector<T> vec) {
    std::cout << "[ ";
    for (auto t : vec) std::cout << t << " ";
    std::cout << " ]\n";
}

ctxt_bit secure_eq(ctxt a, ctxt b, EncInfo& info, std::vector<helib::zzX> * unpackSlotEncoding) {
    ctxt_bit scratch = encrypt_vector(info, ptxt_vec());
    ctxt_bit lt(scratch), gt(scratch);

    helib::compareTwoNumbers(gt, lt, 
                        helib::CtPtrs_vectorCt(a), 
                        helib::CtPtrs_vectorCt(b), true, unpackSlotEncoding);
        lt += gt;
        lt.addConstant(NTL::ZZX(1));

    return lt;
}

ctxt secure_index(std::vector<ctxt> arr, ctxt idx, EncInfo& info, std::vector<helib::zzX> * unpackSlotEncoding) {
    ctxt result = encrypt(info, ptxt{0});
    ctxt zero = encrypt(info, ptxt{0});
    ctxt_bit scratch = encrypt_vector(info, ptxt_vec());
    ctxt_bit lt(scratch), gt(scratch);
    for (int i = 0; i < arr.size(); i++) {
        
        ctxt cidx = encrypt(info, ptxt{i});
        helib::compareTwoNumbers(gt, lt, 
                        helib::CtPtrs_vectorCt(idx), 
                        helib::CtPtrs_vectorCt(cidx), true, unpackSlotEncoding);
        lt += gt;
        lt.addConstant(NTL::ZZX(1));

        ctxt masked = mux(gt, lt, zero, arr[i]);
        // std::cout << "masked = " << decrypt(info, masked)[0] << "\n";
        // std::cout << "arr[i] = " << decrypt(info, arr[i])[0] << "\n";
        result = add(result, masked, unpackSlotEncoding);
    }
    // std::cout << "arr[" << decrypt(info, idx)[0] << "]: " << decrypt(info, result)[0] << "\n";
    return result;
}