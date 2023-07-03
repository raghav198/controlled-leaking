
#include <helib/FHE.h>

#include "mux-common.hpp"

#ifdef DEBUG
#define show(x) _show(info, x, #x, 11)
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
    return std::vector<int>{8, 6};
}

compute_data Prep(EncInfo & info, std::unordered_map<std::string, int> inputs)
{

    compute_data data;
    
    ptxt t0{1, 0, 0, 0, 0, 1, 0, 0, 0};
    data.plaintexts.push_back(encode_vector(info, t0));
    ptxt t1{inputs["input#3.1"], inputs["input#3.0"], 0, 0, inputs["input#3.1"], inputs["input#3.0"], inputs["input#3.1"], inputs["input#3.0"], 0};
    data.ciphertexts.push_back(encrypt_vector(info, t1));
    ptxt t2{0, 1, 0, 0, 0, 1, 0, 1, 0};
    data.plaintexts.push_back(encode_vector(info, t2));
    ptxt t3{1, 0, 0, 0, 1, 0, 1, 0, 0};
    data.plaintexts.push_back(encode_vector(info, t3));
    ptxt t4{inputs["input#2.1"], 0, 0, 0, inputs["input#1.1"], 0, inputs["input#0.0"], 0, 0};
    data.ciphertexts.push_back(encrypt_vector(info, t4));
    ptxt t5{0, 0, 0, 0, 1, 0, 1, 0, 0};
    data.plaintexts.push_back(encode_vector(info, t5));
    ptxt t6{inputs["input#2.0"], 0, 0, 0, inputs["input#1.0"], 0, inputs["input#0.1"], 0, 0};
    data.ciphertexts.push_back(encrypt_vector(info, t6));
    ptxt t7{0, 0, 0, 0, 1, 0, 1, 0, 0};
    data.plaintexts.push_back(encode_vector(info, t7));
    ptxt t8{1, 0, 1, 0, 0, 0, 0, 0, 0};
    data.plaintexts.push_back(encode_vector(info, t8));
    ptxt t10{0, 0, 0, 0, 0, 0, 0, 0, 0};
    data.plaintexts.push_back(encode_vector(info, t10));
    data.masks["100000000"] = make_mask(info, "100000000");
    data.masks["000000001"] = make_mask(info, "000000001");
    data.masks["001000000"] = make_mask(info, "001000000");
    data.masks["000000100"] = make_mask(info, "000000100");
    return data;
}

ctxt_bit Compute(EncInfo & info, compute_data data)
{
    ctxt_bit v0 = data.ciphertexts[0];
    v0 += data.plaintexts[0];
    show(v0);
    ctxt_bit v1 = v0;
    v1 += data.plaintexts[1];
    show(v1);
    ctxt_bit s0 = v1;
    info.context.getEA().rotate(s0, 8);
    ctxt_bit v2 = v0;
    v2 += data.plaintexts[2];
    show(v2);
    ctxt_bit v3 = s0;
    v3.multiplyBy(v2);
    show(v3);
    ctxt_bit s1 = v3;
    info.context.getEA().rotate(s1, 2);
    ctxt_bit v4 = v3;
    v4.multiplyBy(data.ciphertexts[1]);
    show(v4);
    ctxt_bit s2 = v4;
    info.context.getEA().rotate(s2, 2);
    ctxt_bit v5 = v3;
    v5 += data.plaintexts[3];
    show(v5);
    ctxt_bit s3 = v5;
    info.context.getEA().rotate(s3, 2);
    ctxt_bit v6 = v3;
    v6.multiplyBy(data.ciphertexts[2]);
    show(v6);
    ctxt_bit s4 = v6;
    info.context.getEA().rotate(s4, 2);
    ctxt_bit s5 = v6;
    info.context.getEA().rotate(s5, 4);
    ctxt_bit v7 = v3;
    v7 += data.plaintexts[4];
    show(v7);
    ctxt_bit s6 = v7;
    info.context.getEA().rotate(s6, 4);
    ctxt_bit t9 = blend_bits({{v3, data.masks["100000000"]}, {s1, data.masks["001000000"]}});
    ctxt_bit v8 = t9;
    v8 += data.plaintexts[5];
    show(v8);
    ctxt_bit v9 = v8;
    v9 *= data.plaintexts[6];
    show(v9);
    ctxt_bit t11 = blend_bits({{v4, data.masks["100000000"]}, {s4, data.masks["001000000"]}});
    ctxt_bit v10 = t11;
    v10 += v9;
    show(v10);
    ctxt_bit s7 = v10;
    info.context.getEA().rotate(s7, 6);
    ctxt_bit t12 = blend_bits({{s3, data.masks["000000100"]}, {s6, data.masks["000000001"]}});
    ctxt_bit v11 = t12;
    v11.multiplyBy(s7);
    show(v11);
    ctxt_bit t13 = blend_bits({{s2, data.masks["000000100"]}, {s5, data.masks["000000001"]}});
    ctxt_bit v12 = t13;
    v12 += v11;
    show(v12);
    ctxt_bit t14 = blend_bits({{v7, data.masks["000000100"]}, {s3, data.masks["000000001"]}});
    ctxt_bit v13 = t14;
    v13.multiplyBy(v12);
    show(v13);
    ctxt_bit t15 = blend_bits({{v6, data.masks["000000100"]}, {s2, data.masks["000000001"]}});
    ctxt_bit v14 = t15;
    v14 += v13;
    show(v14);
    return v14;
}
    