#include <coyote-runtime.hpp>

zzx_vec make_mask(EncInfo& info, std::string data, int positive_pads, int negative_pads) {
    std::vector<long> mask_data;
    for (auto c : data)
        mask_data.push_back(c - '0');
    
    mask_data = pad_vector(mask_data, positive_pads, negative_pads, info.nslots);

    return zzx_vec(info.context, mask_data);
}

ctxt truncate(ctxt val, int width) {
    if (val.size() > width) {
        val.erase(val.begin() + width, val.end());
        // val.resize(width);
    }
    return val;
}

ctxt add(ctxt a, ctxt b, std::vector<helib::zzX> * unpackSlotEncoding) {
    assert(a.size() == b.size());
    ctxt result;
    helib::CtPtrs_vectorCt result_wrapped(result);
    helib::addTwoNumbers(result_wrapped, helib::CtPtrs_vectorCt(a), helib::CtPtrs_vectorCt(b), 0L, unpackSlotEncoding);
    return truncate(result, a.size());
}

ctxt sub(ctxt a, ctxt b, std::vector<helib::zzX> * unpackSlotEncoding) {
    assert(a.size() == b.size());
    ctxt result(a);
    helib::CtPtrs_vectorCt result_wrapped(result);
    helib::subtractBinary(result_wrapped, helib::CtPtrs_vectorCt(a), helib::CtPtrs_vectorCt(b), unpackSlotEncoding);
    return truncate(result, a.size());
}

ctxt mul(ctxt a, ctxt b, std::vector<helib::zzX> * unpackSlotEncoding) {
    assert(a.size() == b.size());
    ctxt result;
    helib::CtPtrs_vectorCt result_wrapped(result);
    helib::multTwoNumbers(result_wrapped, helib::CtPtrs_vectorCt(a), helib::CtPtrs_vectorCt(b), false, 0, unpackSlotEncoding);
    return truncate(result, a.size());
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

    // relinearize once at the end
    for (auto & bit : result)
        bit.reLinearize();
    
    return result;
}


ctxt_bit blend_bits(std::initializer_list<std::pair<ctxt_bit, helib::Ptxt<helib::BGV>>> sources) {
    std::vector<ctxt_bit> masked_sources;
    for (auto source : sources) {
        ctxt_bit masked_bit(source.first);
        masked_bit *= source.second;
        masked_sources.push_back(masked_bit);
    }

    ctxt_bit result(masked_sources[0]);
    for (int i = 1; i < masked_sources.size(); i++)
        result += masked_sources[i];

    result.reLinearize();
    
    return result;
}

ctxt encrypt(EncInfo& info, ptxt n, int positive_pads, int negative_pads) {
#ifdef DEBUG
    std::cout << "Encrypting: ";
    for (auto val : n) std::cout << val << " ";
    std::cout << "\n";
#endif
    auto bits = decompose_bits(n, bitwidth);
    ctxt output;
    for (auto bit : bits) {
        output.push_back(encrypt_vector(info, bit, positive_pads, negative_pads));
    }
    return output;
}

ptxt decrypt(EncInfo& info, ctxt c) {
    ptxt result;
    helib::decryptBinaryNums(result, helib::CtPtrs_vectorCt(c), *info.sk, info.context.getEA(), true);
    return result;
}