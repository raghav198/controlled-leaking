#pragma once

#include <memory>

#include "data-owner.hpp"
#include "model-owner.hpp"
#include "vectrees.hpp"

class SallyServer
{

  public:
    std::shared_ptr<DataOwner> danielle;
    std::shared_ptr<ModelOwner> maurice;
    const helib::EncryptedArray& ea;

    SallyServer(std::shared_ptr<DataOwner> d, std::shared_ptr<ModelOwner> m,
                const helib::EncryptedArray& ea)
        : danielle(d), maurice(m), ea(ea), model(nullptr)
    {
    }

    void LoadModel();
    void ExecuteQuery();

    CtxtModelDescription *model;
};

std::vector<ctxt_vec> prefix_mult(std::vector<ctxt_vec> vals);
ctxt_vec compare(std::vector<ctxt_vec> thresholds, std::vector<ctxt_vec> features);

ctxt_vec transform_vec(ctxt_mat matrix, ctxt_vec vec);