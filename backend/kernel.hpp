#ifndef KERNEL_HPP
#define KERNEL_HPP

#include "coyote-runtime.hpp"

#include <vector>
#include <string>
#include <copse/model-owner.hpp>

enum class kernel {
    left,
    right,
    label
};

struct client_data {
    std::unordered_map<kernel, std::vector<ctxt>> input_wires;
    std::unordered_map<std::string, helib::Ptxt<helib::BGV>> masks;
    CtxtModelDescription model;
};

struct CoyoteKernel {
    EncInfo& info;
    int width;

    CoyoteKernel(EncInfo& info, int width) : info(info), width(width) {}

    std::vector<ctxt> input_wires, output_wires;
    std::unordered_map<std::string, helib::Ptxt<helib::BGV>> masks;

    virtual void Prepare(std::unordered_map<std::string, int>) = 0; // populate input_wires from values assigned to variables and populate masks
    virtual void Compute() = 0; // populate output_wires from input_wires

    void add_masks(std::initializer_list<std::string> mask_list) {
        for (auto mask : mask_list) {
            masks[mask] = make_mask(info, mask);
        }
    }

    virtual ~CoyoteKernel() {};

    std::vector<int> get_output(int index) {
        ptxt_vec result = decrypt(info, output_wires[index]);
        return std::vector<int>(result.begin(), result.begin() + width);
    }
};

struct COILLabels : public CoyoteKernel {
    COILLabels(EncInfo& info);
    virtual void Prepare(std::unordered_map<std::string, int>) override;
    virtual void Compute() override;
};

struct COILLeftKernel : public CoyoteKernel {
    COILLeftKernel(EncInfo& info);
    virtual void Prepare(std::unordered_map<std::string, int>) override;
    virtual void Compute() override;
};

struct COILRightKernel : public CoyoteKernel {
    COILRightKernel(EncInfo& info);
    virtual void Prepare(std::unordered_map<std::string, int>) override;
    virtual void Compute() override;
};

class COILMaurice : public ModelOwner {
public:
    std::string eq_mask();
    std::string lt_mask();
    COILMaurice(EncInfo& info) : ModelOwner(info) {}
    virtual PtxtModelDescription plaintext_model();
    virtual CtxtModelDescription* GenerateModel();
};

#endif