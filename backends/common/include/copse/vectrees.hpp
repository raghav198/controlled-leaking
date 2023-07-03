#ifndef COPSE_VECTREES_HPP
#define COPSE_VECTREES_HPP

#include <NTL/BasicThreadPool.h>
#include <chrono>
#include <helib/FHE.h>
#include <vector>

// #define PTXT_MODEL
// #define PTXT_DATA

using ptxt_vec = std::vector<long>;
using ptxt_mat = std::vector<ptxt_vec>;

using ctxt_vec = helib::Ctxt;
using ctxt_mat = std::vector<ctxt_vec>;

using zzx_vec = helib::Ptxt<helib::BGV>;
using zzx_mat = std::vector<zzx_vec>;

#ifdef PTXT_MODEL
#define model_vec zzx_vec
#define model_mat zzx_mat
#else
#define model_vec ctxt_vec
#define model_mat ctxt_mat
#endif

#if defined(PTXT_DATA) && defined(PTXT_MODEL)
#define data_vec zzx_vec
#else
#define data_vec ctxt_vec
#endif

#ifndef VECTREE_THREADED
#undef NTL_EXEC_INDEX
#undef NTL_EXEC_INDEX_END
#define NTL_EXEC_INDEX(N, i) for (int i = 0; i < (N); i++)
#define NTL_EXEC_INDEX_END
#endif


struct EncInfo
{
    helib::Context context;
    helib::SecKey *sk;
    long nslots;
    EncInfo(int m, int p, int r, int bits, int cols) : 
        EncInfo(helib::ContextBuilder<helib::BGV>().m(m).p(p).r(r).bits(bits).c(cols)) {}

    EncInfo(helib::ContextBuilder<helib::BGV> builder) : context(builder.build())
    {
        sk = new helib::SecKey(context);
        sk->GenSecKey();
        helib::addSome1DMatrices(*sk);
        helib::addSomeFrbMatrices(*sk);
        nslots = context.getEA().size();
    }

};

template <typename Unit> class Timer
{
    using clock = std::chrono::high_resolution_clock;

  private:
    std::vector<Unit> &acc;
    decltype(clock::now()) t1, t2;

  public:
    Timer(std::vector<Unit> &acc) : acc(acc)
    {
        t1 = clock::now();
    }
    void measure()
    {
        t2 = clock::now();
        acc.push_back(std::chrono::duration_cast<Unit>(t2 - t1));
        t1 = clock::now();
    }
};

ptxt_vec pad_vector(ptxt_vec vec, int positive_pads, int negative_pads, int nslots);
std::vector<ptxt_vec> decompose_bits(ptxt_vec values, int bitwidth);
ctxt_vec encrypt_vector(const EncInfo &, ptxt_vec, int positive_pads = 1, int negative_pads = 1);
ptxt_vec decrypt_vector(const EncInfo &, ctxt_vec);
zzx_vec encode_vector(const EncInfo &, ptxt_vec, int positive_pads = 1, int negative_pads = 1);
ptxt_vec decode_vector(const EncInfo &, zzx_vec);
ctxt_mat encrypt_matrix(const EncInfo &, ptxt_mat);
zzx_mat encode_matrix(const EncInfo &, ptxt_mat);
std::vector<zzx_vec> generate_rotations(zzx_vec, int, const helib::EncryptedArray&);
std::vector<ctxt_vec> generate_rotations(ctxt_vec, int, const helib::EncryptedArray&);

ctxt_vec mat_mul(zzx_mat, std::vector<ctxt_vec>);
ctxt_vec mat_mul(ctxt_mat, std::vector<ctxt_vec>);
zzx_vec mat_mul(zzx_mat, std::vector<zzx_vec>);
void print_vec(std::ostream &, ptxt_vec);

#endif