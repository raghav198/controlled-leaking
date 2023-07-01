#pragma once

#include "vectrees.hpp"

struct CtxtFeatureVector
{
    std::vector<data_vec> bitwise_features;
};

class DataOwner
{
  protected:
    CtxtFeatureVector encrypt(ptxt_vec, int, int);
    virtual ptxt_vec plaintext_features() = 0;
    const EncInfo &info;

  public:
    friend class SallyServer;
    DataOwner(const EncInfo &info) : info(info)
    {
    }
    CtxtFeatureVector GetFeatureVector(int, int);
    virtual void accept_inference(ctxt_vec) = 0;
    virtual void accept_inference(zzx_vec) = 0;
};