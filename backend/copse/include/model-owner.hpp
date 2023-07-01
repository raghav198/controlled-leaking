#pragma once

#include "vectrees.hpp"

struct PtxtModelDescription
{
    int k;
    int bits = 8;
    std::string name;
    ptxt_vec thresholds;
    ptxt_mat d2b;
    std::vector<ptxt_mat> level_b2s;
    std::vector<ptxt_vec> level_mask;
};

struct CtxtModelDescription
{
    // #ifdef PTXT_MODEL
    //     using vectype = zzx_vec;
    //     using mattype = zzx_mat;
    // #else
    //     using vectype = ctxt_vec;
    //     using mattype = ctxt_mat;
    // #endif
    int k;
    int bits;

    int depth, width, quantized, slots;
    std::string name;

    std::vector<model_vec> thresholds;
    model_mat d2b;
    std::vector<model_mat> level_b2s;
    std::vector<model_vec> level_mask;
};

class ModelOwner
{
  protected:
    CtxtModelDescription *encrypt(PtxtModelDescription);
    virtual PtxtModelDescription plaintext_model() = 0;
    const EncInfo &info;

  public:
    CtxtModelDescription *GenerateModel();
    ModelOwner(const EncInfo &info) : info(info)
    {
    }
};