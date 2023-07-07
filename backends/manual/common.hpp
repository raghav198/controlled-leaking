#ifndef COMMON_HPP
#define COMMON_HPP

#include <helib/binaryCompare.h>
#include "kernel.hpp"

ctxt get_ctxt(EncInfo & info, long val);

ctxt mux(ctxt_bit l_bit, ctxt_bit r_bit, ctxt left, ctxt right);

template<typename T>
void print_vector(std::vector<T> vec);

ctxt_bit secure_eq(ctxt a, ctxt b, EncInfo& info, std::vector<helib::zzX> * unpackSlotEncoding = nullptr);
ctxt secure_index(std::vector<ctxt> arr, ctxt idx, EncInfo& info, std::vector<helib::zzX> * unpackSlotEncoding = nullptr);

#endif