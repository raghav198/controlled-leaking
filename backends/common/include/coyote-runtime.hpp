#ifndef COYOTE_RUNTIME_HPP
#define COYOTE_RUNTIME_HPP

#include <copse/vectrees.hpp>
#include <helib/FHE.h>
#include <helib/binaryArith.h>

#include <vector>

using ctxt = std::vector<helib::Ctxt>;
using ctxt_bit = helib::Ctxt;
using ptxt = std::vector<long>;

#ifdef BITWIDTH
constexpr int bitwidth = BITWIDTH;
#else
constexpr int bitwidth = 8;
#endif

zzx_vec make_mask(EncInfo& info, std::string data, int positive_pads = 1, int negative_pads = 1);
ctxt truncate(ctxt val, int width);

ctxt add(ctxt a, ctxt b, std::vector<helib::zzX> * unpackSlotEncoding = nullptr);
ctxt sub(ctxt a, ctxt b, std::vector<helib::zzX> * unpackSlotEncoding = nullptr);
ctxt mul(ctxt a, ctxt b, std::vector<helib::zzX> * unpackSlotEncoding = nullptr);

ctxt rotate(ctxt val, int amount);
ctxt blend(std::initializer_list<std::pair<ctxt, helib::Ptxt<helib::BGV>>> sources);
ctxt_bit blend_bits(std::initializer_list<std::pair<ctxt_bit, helib::Ptxt<helib::BGV>>> sources);

ctxt encrypt(EncInfo& info, ptxt n, int positive_pads = 1, int negative_pads = 1);
ptxt decrypt(EncInfo& info, ctxt c);

#endif