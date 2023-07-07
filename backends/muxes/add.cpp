
#include <helib/FHE.h>

#include "mux-common.hpp"

#ifdef DEBUG
#define show(x) _show(info, x, #x, 64)
#else
#define show(x)
#endif

void _show(EncInfo & info, ctxt_bit vec, std::string name, int size)
{
    std::cout << name << ": ";
    auto decrypted = decrypt_vector(info, vec);
    for (int i = 0; i < size; i++) {
        std::cout << decrypted[i];
    }
    std::cout << "\n";
}

std::vector<int> lanes()
{
    return std::vector<int>{48, 27, 15, 16, 36, 29, 37, 56};
}

compute_data Prep(EncInfo & info, std::unordered_map<std::string, int> inputs)
{

    compute_data data;
    int pad_count = 1;
    
    ptxt t0{0, inputs["input#1.1"], 0, inputs["input#1.5"], 0, 0, 0, 0, 0, 0, 0, 0, inputs["input#1.2"], 0, inputs["input#1.0"], 0, inputs["input#1.7"], inputs["input#1.6"], inputs["input#1.2"], 0, inputs["input#1.0"], 0, 0, inputs["input#1.3"], inputs["input#1.5"], 0, 0, 0, inputs["input#1.1"], inputs["input#1.1"], 0, 0, inputs["input#1.4"], inputs["input#1.2"], inputs["input#1.5"], 0, inputs["input#1.6"], 0, inputs["input#1.1"], 0, inputs["input#1.4"], 0, 0, 0, inputs["input#1.3"], inputs["input#1.3"], inputs["input#1.1"], inputs["input#1.2"], 0, inputs["input#1.2"], inputs["input#1.0"], inputs["input#1.3"], inputs["input#1.2"], 0, 0, 0, inputs["input#1.4"], 0, inputs["input#1.3"], 0, inputs["input#1.1"], inputs["input#1.4"], inputs["input#1.1"], 0};
    data.ciphertexts.push_back(encrypt_vector(info, t0, pad_count, 1));
    ptxt t1{0, inputs["input#2.1"], 0, inputs["input#2.5"], 0, 0, 0, 0, 0, 0, 0, 0, inputs["input#2.2"], 0, inputs["input#2.0"], 0, inputs["input#2.7"], inputs["input#2.6"], inputs["input#2.2"], 0, inputs["input#2.0"], 0, 0, inputs["input#2.3"], inputs["input#2.5"], 0, 0, 0, inputs["input#2.1"], inputs["input#2.1"], 0, 0, inputs["input#2.4"], inputs["input#2.2"], inputs["input#2.5"], 0, inputs["input#2.6"], 0, inputs["input#2.1"], 0, inputs["input#2.4"], 0, 0, 0, inputs["input#2.3"], inputs["input#2.3"], inputs["input#2.1"], inputs["input#2.2"], 0, inputs["input#2.2"], inputs["input#2.0"], inputs["input#2.3"], inputs["input#2.2"], 0, 0, 0, inputs["input#2.4"], 0, inputs["input#2.3"], 0, inputs["input#2.1"], inputs["input#2.4"], inputs["input#2.1"], 0};
    data.ciphertexts.push_back(encrypt_vector(info, t1, pad_count, 1));
    ptxt t2{inputs["input#1.1"], 0, inputs["input#1.1"], 0, inputs["input#1.0"], inputs["input#1.4"], inputs["input#1.1"], inputs["input#1.1"], 0, inputs["input#1.2"], inputs["input#1.2"], inputs["input#1.0"], 0, 0, 0, inputs["input#1.2"], inputs["input#1.0"], 0, 0, inputs["input#1.0"], 0, inputs["input#1.4"], inputs["input#1.4"], 0, 0, 0, inputs["input#1.3"], inputs["input#1.5"], 0, 0, inputs["input#1.5"], inputs["input#1.6"], 0, 0, 0, inputs["input#1.2"], 0, inputs["input#1.3"], 0, inputs["input#1.3"], 0, inputs["input#1.1"], inputs["input#1.0"], inputs["input#1.0"], 0, 0, 0, 0, inputs["input#1.0"], 0, 0, 0, 0, 0, inputs["input#1.3"], 0, 0, 0, 0, inputs["input#1.2"], 0, 0, 0, inputs["input#1.1"]};
    data.ciphertexts.push_back(encrypt_vector(info, t2, pad_count, 1));
    ptxt t3{inputs["input#2.1"], 0, inputs["input#2.1"], 0, inputs["input#2.0"], inputs["input#2.4"], inputs["input#2.1"], inputs["input#2.1"], 0, inputs["input#2.2"], inputs["input#2.2"], inputs["input#2.0"], 0, 0, 0, inputs["input#2.2"], inputs["input#2.0"], 0, 0, inputs["input#2.0"], 0, inputs["input#2.4"], inputs["input#2.4"], 0, 0, 0, inputs["input#2.3"], inputs["input#2.5"], 0, 0, inputs["input#2.5"], inputs["input#2.6"], 0, 0, 0, inputs["input#2.2"], 0, inputs["input#2.3"], 0, inputs["input#2.3"], 0, inputs["input#2.1"], inputs["input#2.0"], inputs["input#2.0"], 0, 0, 0, 0, inputs["input#2.0"], 0, 0, 0, 0, 0, inputs["input#2.3"], 0, 0, 0, 0, inputs["input#2.2"], 0, 0, 0, inputs["input#2.1"]};
    data.ciphertexts.push_back(encrypt_vector(info, t3, pad_count, 1));
    ptxt t4{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, inputs["input#1.0"], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, inputs["input#1.0"], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, inputs["input#1.0"], 0, inputs["input#1.0"], 0, 0, 0, 0, 0, 0};
    data.ciphertexts.push_back(encrypt_vector(info, t4, pad_count, 1));
    ptxt t5{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, inputs["input#2.0"], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, inputs["input#2.0"], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, inputs["input#2.0"], 0, inputs["input#2.0"], 0, 0, 0, 0, 0, 0};
    data.ciphertexts.push_back(encrypt_vector(info, t5, pad_count, 1));
    ptxt t8{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, inputs["input#0.5"], inputs["input#0.1"], inputs["input#0.4"], 0, 0, 0, inputs["input#0.1"], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, inputs["input#0.5"], 0, 0, inputs["input#0.4"], 0, 0, 0, 0, inputs["input#0.2"], 0, 0, 0, inputs["input#0.2"], 0, 0, inputs["input#0.7"], 0, inputs["input#0.6"], 0, 0, 0, 0, 0, 0, 0, 0};
    data.ciphertexts.push_back(encrypt_vector(info, t8, pad_count, 1));
    ptxt t10{0, 0, 0, 0, 0, 0, 0, 0, inputs["input#1.0"], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    data.ciphertexts.push_back(encrypt_vector(info, t10, pad_count, 1));
    ptxt t11{0, 0, 0, 0, 0, 0, 0, 0, inputs["input#2.0"], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    data.ciphertexts.push_back(encrypt_vector(info, t11, pad_count, 1));
    ptxt t12{inputs["input#0.0"], 0, 0, 0, 0, 0, inputs["input#0.2"], inputs["input#0.2"], 0, 0, 0, inputs["input#0.6"], 0, inputs["input#0.3"], 0, inputs["input#0.1"], inputs["input#0.3"], inputs["input#0.3"], inputs["input#0.3"], 0, inputs["input#0.0"], 0, 0, 0, inputs["input#0.2"], inputs["input#0.4"], 0, 0, 0, 0, 0, 0, 0, inputs["input#0.5"], 0, 0, inputs["input#0.1"], inputs["input#0.3"], 0, 0, inputs["input#0.0"], inputs["input#0.1"], inputs["input#0.2"], inputs["input#0.1"], 0, 0, 0, 0, inputs["input#0.0"], 0, 0, inputs["input#0.4"], inputs["input#0.0"], 0, 0, 0, inputs["input#0.0"], 0, 0, inputs["input#0.0"], 0, 0, inputs["input#0.1"], 0};
    data.ciphertexts.push_back(encrypt_vector(info, t12, pad_count, 1));
    ptxt t20{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, inputs["input#0.0"], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    data.ciphertexts.push_back(encrypt_vector(info, t20, pad_count, 1));
    data.masks["0000000000000000000000000000000000000000000000000000010000000000"] = make_mask(info, "0000000000000000000000000000000000000000000000000000010000000000", pad_count, 1);
    data.masks["0100000000010000000010000000000001000000000000000000000000000000"] = make_mask(info, "0100000000010000000010000000000001000000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000000010000000100000000000000000000000000"] = make_mask(info, "0000000000000000000000000000010000000100000000000000000000000000", pad_count, 1);
    data.masks["0000000100000000000000000000000000000000000000000000000010000000"] = make_mask(info, "0000000100000000000000000000000000000000000000000000000010000000", pad_count, 1);
    data.masks["0000000000001000001000001000000000000000000000010100000000000000"] = make_mask(info, "0000000000001000001000001000000000000000000000010100000000000000", pad_count, 1);
    data.masks["0000000000000000010000000000000000000000010000001000000000000000"] = make_mask(info, "0000000000000000010000000000000000000000010000001000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000001000001000010000000000000000000000000"] = make_mask(info, "0000000000000000000000000001000001000010000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000000100010000000000100000000000"] = make_mask(info, "0000000000000000000000000000000000000100010000000000100000000000", pad_count, 1);
    data.masks["0000000000000000000000000100000000000000000000000100000000000000"] = make_mask(info, "0000000000000000000000000100000000000000000000000100000000000000", pad_count, 1);
    data.masks["0000000000000000010000000100000000000000000000000000000000000000"] = make_mask(info, "0000000000000000010000000100000000000000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000000000000000000000101000000000"] = make_mask(info, "0000000000000000000000000000000000000000000000000000101000000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000001000000000000000000000000000"] = make_mask(info, "0000000000000000000000000000000000001000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000000000000000010000000000000000"] = make_mask(info, "0000000000000000000000000000000000000000000000010000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000001000000000000000000100000000000000000"] = make_mask(info, "0000000000000000000000000001000000000000000000100000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000100000000000000000000000000000000000000"] = make_mask(info, "0000000000000000000000000100000000000000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000000000000000000000000010000000"] = make_mask(info, "0000000000000000000000000000000000000000000000000000000010000000", pad_count, 1);
    data.masks["0000001000010000001000000000000000000000000000000000000000000000"] = make_mask(info, "0000001000010000001000000000000000000000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000000100000000000000000000000000"] = make_mask(info, "0000000000000000000000000000000000000100000000000000000000000000", pad_count, 1);
    data.masks["0000000010000000100000000000100000010000000000000010000000000000"] = make_mask(info, "0000000010000000100000000000100000010000000000000010000000000000", pad_count, 1);
    data.masks["0000000010000000000000000000000000000000000100000000100000000001"] = make_mask(info, "0000000010000000000000000000000000000000000100000000100000000001", pad_count, 1);
    data.masks["0000000000000000000000000000010000001000000001000000000000000000"] = make_mask(info, "0000000000000000000000000000010000001000000001000000000000000000", pad_count, 1);
    data.masks["0000000000001000001000000000000000000001000000000000001000000000"] = make_mask(info, "0000000000001000001000000000000000000001000000000000001000000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000000000000000000000000100000000"] = make_mask(info, "0000000000000000000000000000000000000000000000000000000100000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000000000001000000000000000000000"] = make_mask(info, "0000000000000000000000000000000000000000001000000000000000000000", pad_count, 1);
    data.masks["0000100000000000000000000000000000000000000000000000000000000000"] = make_mask(info, "0000100000000000000000000000000000000000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000000100000000000000000000000000100000011000000"] = make_mask(info, "0000000000000000000000100000000000000000000000000100000011000000", pad_count, 1);
    data.masks["1000000000000000000010000000000000000000000000000000000000000000"] = make_mask(info, "1000000000000000000010000000000000000000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000000100000000000000000000000000000000000000000"] = make_mask(info, "0000000000000000000000100000000000000000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000000000000000000000000000010000"] = make_mask(info, "0000000000000000000000000000000000000000000000000000000000010000", pad_count, 1);
    data.masks["0000000000000010000000000000100000000000000000000000000000000000"] = make_mask(info, "0000000000000010000000000000100000000000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000100000000000000000000000000000000000000000000000"] = make_mask(info, "0000000000000000100000000000000000000000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000010010000000000000000000000000000000000000000000000"] = make_mask(info, "0000000000000010010000000000000000000000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000010000000000000000000000000000000000000"] = make_mask(info, "0000000000000000000000000010000000000000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000000001000110000100000000001000010000000000000"] = make_mask(info, "0000000000000000000000001000110000100000000001000010000000000000", pad_count, 1);
    data.masks["0000000000000000000000000001000000000000000000000000000000000000"] = make_mask(info, "0000000000000000000000000001000000000000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000000000000000000100000000000000"] = make_mask(info, "0000000000000000000000000000000000000000000000000100000000000000", pad_count, 1);
    data.masks["0000000000000000000010000000000000000010000000000000000000000000"] = make_mask(info, "0000000000000000000010000000000000000010000000000000000000000000", pad_count, 1);
    data.masks["0000000010000000000000000000000000000000000000000100000000000000"] = make_mask(info, "0000000010000000000000000000000000000000000000000100000000000000", pad_count, 1);
    data.masks["0000000000000000000000010000000000100000000000100000000000000000"] = make_mask(info, "0000000000000000000000010000000000100000000000100000000000000000", pad_count, 1);
    data.masks["0000000000000000000010000000000000000001000100000000000000000000"] = make_mask(info, "0000000000000000000010000000000000000001000100000000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000000000000000000010000000000000"] = make_mask(info, "0000000000000000000000000000000000000000000000000010000000000000", pad_count, 1);
    data.masks["0000000000000001100000000001010000001100000000000000000010000000"] = make_mask(info, "0000000000000001100000000001010000001100000000000000000010000000", pad_count, 1);
    data.masks["0000000000000100000000000000000000000000000000000000000000000000"] = make_mask(info, "0000000000000100000000000000000000000000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000100000001000000000000000100000000001000000000000"] = make_mask(info, "0000000000000000100000001000000000000000100000000001000000000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000000000000000000000000000000001"] = make_mask(info, "0000000000000000000000000000000000000000000000000000000000000001", pad_count, 1);
    data.masks["0000000000000000000000010000000000010010000000000000000000000000"] = make_mask(info, "0000000000000000000000010000000000010010000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000000000001000000000100000000000000000000"] = make_mask(info, "0000000000000000000000000000000001000000000100000000000000000000", pad_count, 1);
    data.masks["0000000000000001000000000000000000000000000000000000000000000000"] = make_mask(info, "0000000000000001000000000000000000000000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000010000000000000000000000000000000000000000000000"] = make_mask(info, "0000000000000000010000000000000000000000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000000001000000000000000000000000"] = make_mask(info, "0000000000000000000000000000000000000001000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000000000000000100000000000000000"] = make_mask(info, "0000000000000000000000000000000000000000000000100000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000000010000000000000000000000000"] = make_mask(info, "0000000000000000000000000000000000000010000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000001000000000000000000000000000000000000000000"] = make_mask(info, "0000000000000000000001000000000000000000000000000000000000000000", pad_count, 1);
    data.masks["0100000000000000000000000000000000000000000000000000000000000000"] = make_mask(info, "0100000000000000000000000000000000000000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000010000000100000000000000000000000000000000000"] = make_mask(info, "0000000000000000000010000000100000000000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000001000000000000000000000000000000000000000010000000"] = make_mask(info, "0000000000000001000000000000000000000000000000000000000010000000", pad_count, 1);
    data.masks["0000000000000000000010000000000000000000000000000000000000000000"] = make_mask(info, "0000000000000000000010000000000000000000000000000000000000000000", pad_count, 1);
    data.masks["0000100000000000000000100000000000000000000000000000000001000000"] = make_mask(info, "0000100000000000000000100000000000000000000000000000000001000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000000010000000000100000000000000"] = make_mask(info, "0000000000000000000000000000000000000010000000000100000000000000", pad_count, 1);
    data.masks["0000000000000100000000000100000000000000000000000000000101000000"] = make_mask(info, "0000000000000100000000000100000000000000000000000000000101000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000000000010000000000000000000000"] = make_mask(info, "0000000000000000000000000000000000000000010000000000000000000000", pad_count, 1);
    data.masks["0000000010000000000000000000000000000000000000000000000000000000"] = make_mask(info, "0000000010000000000000000000000000000000000000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000001000000000001000000000000010"] = make_mask(info, "0000000000000000000000000000000000001000000000001000000000000010", pad_count, 1);
    data.masks["0000000000000000000000000000000000000000100000000000000000000000"] = make_mask(info, "0000000000000000000000000000000000000000100000000000000000000000", pad_count, 1);
    data.masks["0000000000000000000000000000000000000000000000001000000000000000"] = make_mask(info, "0000000000000000000000000000000000000000000000001000000000000000", pad_count, 1);
    data.masks["0000000000010000000000000000000000000000100000000000000000000000"] = make_mask(info, "0000000000010000000000000000000000000000100000000000000000000000", pad_count, 1);
    return data;
}

ctxt_bit Compute(EncInfo & info, compute_data data)
{
    ctxt_bit v0 = data.ciphertexts[0];
    v0 += data.ciphertexts[1];
    show(v0);
    ctxt_bit s0 = v0;
    info.context.getEA().rotate(s0, -30);
    show(s0);
    ctxt_bit s1 = v0;
    info.context.getEA().rotate(s1, 8);
    show(s1);
    ctxt_bit s2 = v0;
    info.context.getEA().rotate(s2, 3);
    show(s2);
    ctxt_bit s3 = v0;
    info.context.getEA().rotate(s3, 4);
    show(s3);
    ctxt_bit s4 = v0;
    info.context.getEA().rotate(s4, -27);
    show(s4);
    ctxt_bit s5 = v0;
    info.context.getEA().rotate(s5, -10);
    show(s5);
    ctxt_bit s6 = v0;
    info.context.getEA().rotate(s6, -16);
    show(s6);
    ctxt_bit s7 = v0;
    info.context.getEA().rotate(s7, 12);
    show(s7);
    ctxt_bit s8 = v0;
    info.context.getEA().rotate(s8, -8);
    show(s8);
    ctxt_bit v1 = data.ciphertexts[2];
    v1.multiplyBy(data.ciphertexts[3]);
    show(v1);
    ctxt_bit s9 = v1;
    info.context.getEA().rotate(s9, 12);
    show(s9);
    ctxt_bit s10 = v1;
    info.context.getEA().rotate(s10, -30);
    show(s10);
    ctxt_bit s11 = v1;
    info.context.getEA().rotate(s11, 4);
    show(s11);
    ctxt_bit s12 = v1;
    info.context.getEA().rotate(s12, 19);
    show(s12);
    ctxt_bit s13 = v1;
    info.context.getEA().rotate(s13, 16);
    show(s13);
    ctxt_bit s14 = v1;
    info.context.getEA().rotate(s14, -10);
    show(s14);
    ctxt_bit s15 = v1;
    info.context.getEA().rotate(s15, -8);
    show(s15);
    ctxt_bit s16 = v1;
    info.context.getEA().rotate(s16, 3);
    show(s16);
    ctxt_bit s17 = v1;
    info.context.getEA().rotate(s17, 8);
    show(s17);
    ctxt_bit s18 = v1;
    info.context.getEA().rotate(s18, -16);
    show(s18);
    ctxt_bit t6 = blend_bits({{s4, data.masks["0100000000000000000000000000000000000000000000000000000000000000"]}, {s0, data.masks["0000000010000000100000000000100000010000000000000010000000000000"]}, {s1, data.masks["0000000000010000000000000000000000000000100000000000000000000000"]}, {v0, data.masks["0000000000001000001000001000000000000000000000010100000000000000"]}, {s2, data.masks["0000000000000000000010000000000000000001000100000000000000000000"]}, {s5, data.masks["0000000000000000000000010000000000100000000000100000000000000000"]}, {s3, data.masks["0000000000000000000000000001000001000010000000000000000000000000"]}, {s6, data.masks["0000000000000000000000000000010000001000000001000000000000000000"]}, {s8, data.masks["0000000000000000000000000000000000000000000000000000101000000000"]}, {s7, data.masks["0000000000000000000000000000000000000000000000000000000000000001"]}, {data.ciphertexts[4], data.masks["0000000000000100000000000100000000000000000000000000000101000000"]}});
    ctxt_bit t7 = blend_bits({{s14, data.masks["0100000000010000000010000000000001000000000000000000000000000000"]}, {s11, data.masks["0000000010000000000000000000000000000000000100000000100000000001"]}, {s9, data.masks["0000000000001000001000000000000000000001000000000000001000000000"]}, {v1, data.masks["0000000000000000100000000000000000000000000000000000000000000000"]}, {s13, data.masks["0000000000000000000000010000000000010010000000000000000000000000"]}, {s12, data.masks["0000000000000000000000001000110000100000000001000010000000000000"]}, {s15, data.masks["0000000000000000000000000001000000000000000000100000000000000000"]}, {s10, data.masks["0000000000000000000000000000000000001000000000000000000000000000"]}, {s16, data.masks["0000000000000000000000000000000000000000100000000000000000000000"]}, {s18, data.masks["0000000000000000000000000000000000000000000000010000000000000000"]}, {s17, data.masks["0000000000000000000000000000000000000000000000000100000000000000"]}, {data.ciphertexts[5], data.masks["0000000000000100000000000100000000000000000000000000000101000000"]}});
    ctxt_bit v2 = t6;
    v2 += t7;
    show(v2);
    ctxt_bit s19 = v2;
    info.context.getEA().rotate(s19, 6);
    show(s19);
    ctxt_bit s20 = v2;
    info.context.getEA().rotate(s20, 5);
    show(s20);
    ctxt_bit s21 = v2;
    info.context.getEA().rotate(s21, -5);
    show(s21);
    ctxt_bit s22 = v2;
    info.context.getEA().rotate(s22, -40);
    show(s22);
    ctxt_bit s23 = v2;
    info.context.getEA().rotate(s23, 8);
    show(s23);
    ctxt_bit s24 = v2;
    info.context.getEA().rotate(s24, -44);
    show(s24);
    ctxt_bit s25 = v2;
    info.context.getEA().rotate(s25, 12);
    show(s25);
    ctxt_bit s26 = v2;
    info.context.getEA().rotate(s26, -27);
    show(s26);
    ctxt_bit s27 = v2;
    info.context.getEA().rotate(s27, -12);
    show(s27);
    ctxt_bit s28 = v2;
    info.context.getEA().rotate(s28, -20);
    show(s28);
    ctxt_bit s29 = v2;
    info.context.getEA().rotate(s29, -29);
    show(s29);
    ctxt_bit s30 = v2;
    info.context.getEA().rotate(s30, -16);
    show(s30);
    ctxt_bit t9 = blend_bits({{v0, data.masks["0000000000000000000010000000000000000000000000000000000000000000"]}, {s23, data.masks["0000000000000000000001000000000000000000000000000000000000000000"]}, {s26, data.masks["0000000000000000000000100000000000000000000000000000000000000000"]}, {s19, data.masks["0000000000000000000000000010000000000000000000000000000000000000"]}, {s24, data.masks["0000000000000000000000000000000000000010000000000000000000000000"]}, {s30, data.masks["0000000000000000000000000000000000000000010000000000000000000000"]}, {s25, data.masks["0000000000000000000000000000000000000000000000100000000000000000"]}, {s20, data.masks["0000000000000000000000000000000000000000000000000010000000000000"]}, {s2, data.masks["0000000000000000000000000000000000000000000000000000010000000000"]}, {v2, data.masks["0000000000000000000000000000000000000000000000000000000100000000"]}});
    ctxt_bit v3 = data.ciphertexts[6];
    v3.multiplyBy(t9);
    show(v3);
    ctxt_bit s31 = v3;
    info.context.getEA().rotate(s31, -58);
    show(s31);
    ctxt_bit s32 = v3;
    info.context.getEA().rotate(s32, -5);
    show(s32);
    ctxt_bit s33 = v3;
    info.context.getEA().rotate(s33, -1);
    show(s33);
    ctxt_bit s34 = v3;
    info.context.getEA().rotate(s34, -27);
    show(s34);
    ctxt_bit v4 = data.ciphertexts[7];
    v4 += data.ciphertexts[8];
    show(v4);
    ctxt_bit s35 = v4;
    info.context.getEA().rotate(s35, -30);
    show(s35);
    ctxt_bit t13 = blend_bits({{s26, data.masks["1000000000000000000010000000000000000000000000000000000000000000"]}, {s21, data.masks["0000001000010000001000000000000000000000000000000000000000000000"]}, {s19, data.masks["0000000100000000000000000000000000000000000000000000000010000000"]}, {s20, data.masks["0000000000000100000000000000000000000000000000000000000000000000"]}, {s28, data.masks["0000000000000001000000000000000000000000000000000000000000000000"]}, {s27, data.masks["0000000000000000100000001000000000000000100000000001000000000000"]}, {s29, data.masks["0000000000000000010000000100000000000000000000000000000000000000"]}, {v2, data.masks["0000000000000000000000000000000001000000000100000000000000000000"]}, {s22, data.masks["0000000000000000000000000000000000001000000000001000000000000010"]}, {s25, data.masks["0000000000000000000000000000000000000100010000000000100000000000"]}, {s35, data.masks["0000000000000000000000000000000000000000001000000000000000000000"]}, {s24, data.masks["0000000000000000000000000000000000000000000000000000000000010000"]}});
    ctxt_bit v5 = data.ciphertexts[9];
    v5.multiplyBy(t13);
    show(v5);
    ctxt_bit s36 = v5;
    info.context.getEA().rotate(s36, -27);
    show(s36);
    ctxt_bit s37 = v5;
    info.context.getEA().rotate(s37, -2);
    show(s37);
    ctxt_bit s38 = v5;
    info.context.getEA().rotate(s38, -63);
    show(s38);
    ctxt_bit s39 = v5;
    info.context.getEA().rotate(s39, -13);
    show(s39);
    ctxt_bit s40 = v5;
    info.context.getEA().rotate(s40, -1);
    show(s40);
    ctxt_bit s41 = v5;
    info.context.getEA().rotate(s41, -5);
    show(s41);
    ctxt_bit s42 = v5;
    info.context.getEA().rotate(s42, -29);
    show(s42);
    ctxt_bit s43 = v5;
    info.context.getEA().rotate(s43, -31);
    show(s43);
    ctxt_bit t14 = blend_bits({{s38, data.masks["0000000000000010010000000000000000000000000000000000000000000000"]}, {s41, data.masks["0000000000000000000010000000100000000000000000000000000000000000"]}, {s43, data.masks["0000000000000000000000000100000000000000000000000000000000000000"]}, {s39, data.masks["0000000000000000000000000000000000000010000000000100000000000000"]}, {s37, data.masks["0000000000000000000000000000000000000000010000000000000000000000"]}, {s36, data.masks["0000000000000000000000000000000000000000000000001000000000000000"]}});
    ctxt_bit t15 = blend_bits({{s34, data.masks["0000000000000010000000000000100000000000000000000000000000000000"]}, {s32, data.masks["0000000000000000010000000000000000000000010000001000000000000000"]}, {v3, data.masks["0000000000000000000010000000000000000010000000000000000000000000"]}, {s33, data.masks["0000000000000000000000000100000000000000000000000100000000000000"]}});
    ctxt_bit v6 = t14;
    v6 += t15;
    show(v6);
    ctxt_bit s44 = v6;
    info.context.getEA().rotate(s44, -56);
    show(s44);
    ctxt_bit s45 = v6;
    info.context.getEA().rotate(s45, -3);
    show(s45);
    ctxt_bit s46 = v6;
    info.context.getEA().rotate(s46, -21);
    show(s46);
    ctxt_bit s47 = v6;
    info.context.getEA().rotate(s47, -46);
    show(s47);
    ctxt_bit t16 = blend_bits({{s37, data.masks["0000100000000000000000100000000000000000000000000000000001000000"]}, {s38, data.masks["0000000010000000000000000000000000000000000000000100000000000000"]}, {v5, data.masks["0000000000000001000000000000000000000000000000000000000000000000"]}, {s40, data.masks["0000000000000000010000000000000000000000000000000000000000000000"]}, {s44, data.masks["0000000000000000000000000100000000000000000000000000000000000000"]}, {s36, data.masks["0000000000000000000000000000000000000100000000000000000000000000"]}, {s39, data.masks["0000000000000000000000000000000000000001000000000000000000000000"]}, {s47, data.masks["0000000000000000000000000000000000000000000000000000000010000000"]}});
    ctxt_bit t17 = blend_bits({{s39, data.masks["0000100000000000000000000000000000000000000000000000000000000000"]}, {s42, data.masks["0000000010000000000000000000000000000000000000000000000000000000"]}, {s36, data.masks["0000000000000001000000000000000000000000000000000000000000000000"]}, {s45, data.masks["0000000000000000010000000100000000000000000000000000000000000000"]}, {s44, data.masks["0000000000000000000000100000000000000000000000000100000011000000"]}, {s38, data.masks["0000000000000000000000000000000000000100000000000000000000000000"]}, {s37, data.masks["0000000000000000000000000000000000000001000000000000000000000000"]}});
    ctxt_bit v7 = t16;
    v7 += t17;
    show(v7);
    ctxt_bit s48 = v7;
    info.context.getEA().rotate(s48, -56);
    show(s48);
    ctxt_bit s49 = v7;
    info.context.getEA().rotate(s49, -52);
    show(s49);
    ctxt_bit s50 = v7;
    info.context.getEA().rotate(s50, -50);
    show(s50);
    ctxt_bit s51 = v7;
    info.context.getEA().rotate(s51, -21);
    show(s51);
    ctxt_bit s52 = v7;
    info.context.getEA().rotate(s52, -3);
    show(s52);
    ctxt_bit s53 = v7;
    info.context.getEA().rotate(s53, -20);
    show(s53);
    ctxt_bit v8 = s46;
    v8 += v7;
    show(v8);
    ctxt_bit s54 = v8;
    info.context.getEA().rotate(s54, -12);
    show(s54);
    ctxt_bit t18 = blend_bits({{s41, data.masks["0000000000000001000000000000000000000000000000000000000000000000"]}, {s51, data.masks["0000000000000000100000000000000000000000000000000000000000000000"]}, {s39, data.masks["0000000000000000000000000001000000000000000000000000000000000000"]}, {s53, data.masks["0000000000000000000000000000010000000100000000000000000000000000"]}, {s52, data.masks["0000000000000000000000000000000000001000000000000000000000000000"]}, {s54, data.masks["0000000000000000000000000000000000000000000000000000000010000000"]}});
    ctxt_bit t19 = blend_bits({{v7, data.masks["0000000000000001000000000000000000000000000000000000000010000000"]}, {s48, data.masks["0000000000000000100000000000000000000000000000000000000000000000"]}, {s31, data.masks["0000000000000000000000000001000000000000000000000000000000000000"]}, {s49, data.masks["0000000000000000000000000000010000000100000000000000000000000000"]}, {s50, data.masks["0000000000000000000000000000000000001000000000000000000000000000"]}});
    ctxt_bit v9 = t18;
    v9 += t19;
    show(v9);
    ctxt_bit v10 = data.ciphertexts[10];
    v10.multiplyBy(s0);
    show(v10);
    ctxt_bit v11 = blend_bits({{v9, data.masks["0000000000000001100000000001010000001100000000000000000010000000"]}, {v10, data.masks["0000000000000000000000000000000000000000000000001000000000000000"]}});
    return v11;
}
    