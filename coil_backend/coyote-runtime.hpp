#ifndef COYOTE_RUNTIME_HPP
#define COYOTE_RUNTIME_HPP

#include <copse/vectrees.hpp>
#include <helib/FHE.h>
#include <helib/binaryArith.h>

#include <vector>

using ctxt = std::vector<helib::Ctxt>;
using ctxt_bit = helib::Ctxt;
using ptxt = std::vector<long>;

constexpr int bitwidth = 8;

zzx_vec make_mask(EncInfo& info, std::string data);
ctxt truncate(ctxt val, int width);

ctxt add(ctxt a, ctxt b, std::vector<helib::zzX> * unpackSlotEncoding = nullptr);
ctxt sub(ctxt a, ctxt b, std::vector<helib::zzX> * unpackSlotEncoding = nullptr);
ctxt mul(ctxt a, ctxt b, std::vector<helib::zzX> * unpackSlotEncoding = nullptr);

ctxt rotate(ctxt val, int amount);
ctxt blend(std::initializer_list<std::pair<ctxt, helib::Ptxt<helib::BGV>>> sources);

ctxt encrypt(EncInfo& info, ptxt n);
ptxt decrypt(EncInfo& info, ctxt c);

#endif