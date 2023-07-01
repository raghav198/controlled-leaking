#include "data-owner.hpp"
#include <iostream>

#ifdef PTXT_DATA
#define get_vector encode_vector
#else
#define get_vector encrypt_vector
#endif

CtxtFeatureVector DataOwner::encrypt(ptxt_vec features, int k, int bits)
{

    // std::cerr << "Encrypting feature vector...\n";

    helib::PubKey &pk = *info.sk;
    CtxtFeatureVector ctxt;
    ptxt_vec padded;
    for (auto f : features)
    {
        for (int i = 0; i < k; i++)
            padded.push_back(f);
    }
    // ctxt.bitwise_features.reserve(FXP_BITS);
    std::vector<ptxt_vec> bitvecs = decompose_bits(padded, bits);
    for (int i = 0; i < bits; i++)
    {
        ctxt.bitwise_features.push_back(get_vector(info, bitvecs[i]));
    }

    return ctxt;
}

CtxtFeatureVector DataOwner::GetFeatureVector(int k, int bits)
{
    return encrypt(plaintext_features(), k, bits);
}