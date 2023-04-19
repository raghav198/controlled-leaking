#pragma once

#include <copse/vectrees.hpp>
#include <helib/FHE.h>
#include <helib/binaryArith.h>

#include <vector>

using ctxt = std::vector<helib::Ctxt>;
using ctxt_bit = helib::Ctxt;
using ptxt = std::vector<long>;

constexpr int bitwidth = 8;

ctxt add(ctxt a, ctxt b) {
    ctxt result;
    helib::CtPtrs_vectorCt result_wrapped(result);
    helib::addTwoNumbers(result_wrapped, helib::CtPtrs_vectorCt(a), helib::CtPtrs_vectorCt(b));
    return result;
}

ctxt sub(ctxt a, ctxt b) {
    ctxt result(a);
    helib::CtPtrs_vectorCt result_wrapped(result);
    helib::subtractBinary(result_wrapped, helib::CtPtrs_vectorCt(a), helib::CtPtrs_vectorCt(b));
    return result;
}

ctxt mul(ctxt a, ctxt b) {
    ctxt result;
    helib::CtPtrs_vectorCt result_wrapped(result);
    helib::multTwoNumbers(result_wrapped, helib::CtPtrs_vectorCt(a), helib::CtPtrs_vectorCt(b));
    return result;
}

ctxt rotate(ctxt val, int amount) {
    ctxt result;
    for (auto bit : val) {
        helib::Ctxt rotated_bit(bit);
        helib::rotate(rotated_bit, amount);
        result.push_back(rotated_bit);
    }
    return result;
}

ctxt blend(std::initializer_list<std::pair<ctxt, helib::Ptxt<helib::BGV>>> sources) {
    std::vector<ctxt> masked_sources;
    for (auto source : sources) {
        ctxt masked_source;
        for (auto bit : source.first) {
            helib::Ctxt masked_bit(bit);
            masked_bit *= source.second;
            masked_source.push_back(masked_bit);
        }
        masked_sources.push_back(masked_source);
    }

    ctxt result;
    helib::CtPtrs_vectorCt result_wrapped(result);
    auto wrapped_sources = helib::CtPtrMat_vectorCt(masked_sources);
    helib::addManyNumbers(result_wrapped, wrapped_sources, bitwidth);
    return result;
}

ctxt encrypt(EncInfo& info, ptxt n) {
    auto bits = decompose_bits(n, bitwidth);
    ctxt output;
    for (auto bit : bits) {
        output.push_back(encrypt_vector(info, bit));
    }
    return output;
}

ptxt decrypt(EncInfo& info, ctxt c) {
    ptxt result;
    helib::decryptBinaryNums(result, helib::CtPtrs_vectorCt(c), *info.sk, info.context.getEA(), true);
    return result;
}