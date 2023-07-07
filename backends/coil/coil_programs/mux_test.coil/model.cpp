
#include <copse/model-owner.hpp>

#include "kernel.hpp"

auto mux_test_model() {
    PtxtModelDescription model;
    model.name = "MUX_TEST";
    model.bits = 8;
    model.k = -1;
    model.thresholds = {};
    
    model.level_mask = {
        {0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1, 1},
        {0, 0, 0, 1, 1, 1},
        {0, 0, 1, 1, 1, 1},
        {0, 1, 1, 1, 1, 1}
    };
    
    model.level_b2s = {
       
        {
            {0, 0, 0, 0, 1},
            {0, 0, 0, 1, 0},
            {0, 0, 1, 0, 0},
            {0, 1, 0, 0, 0},
            {1, 0, 0, 0, 0},
            {1, 0, 0, 0, 0}
        },        
        {
            {0, 0, 0, 0, 1},
            {0, 0, 0, 1, 0},
            {0, 0, 1, 0, 0},
            {0, 1, 0, 0, 0},
            {0, 1, 0, 0, 0},
            {0, 1, 0, 0, 0}
        },        
        {
            {0, 0, 0, 0, 1},
            {0, 0, 0, 1, 0},
            {0, 0, 1, 0, 0},
            {0, 0, 1, 0, 0},
            {0, 0, 1, 0, 0},
            {0, 0, 1, 0, 0}
        },        
        {
            {0, 0, 0, 0, 1},
            {0, 0, 0, 1, 0},
            {0, 0, 0, 1, 0},
            {0, 0, 0, 1, 0},
            {0, 0, 0, 1, 0},
            {0, 0, 0, 1, 0}
        },        
        {
            {0, 0, 0, 0, 1},
            {0, 0, 0, 0, 1},
            {0, 0, 0, 0, 1},
            {0, 0, 0, 0, 1},
            {0, 0, 0, 0, 1},
            {0, 0, 0, 0, 1}
        } 
    };
    
    return model;
}

std::string COILMaurice::eq_mask() { return "11111"; } 
std::string COILMaurice::lt_mask() { return "00000"; }
PtxtModelDescription COILMaurice::plaintext_model() {
    return mux_test_model();
}
CtxtModelDescription* COILMaurice::GenerateModel() {
    return ModelOwner::GenerateModel();
}
