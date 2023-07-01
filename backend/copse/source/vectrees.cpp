#include "vectrees.hpp"

#include <iterator>

// return a vector of generalized diagonals of 'mat'
ptxt_mat get_diagonals(ptxt_mat mat)
{
    // std::cerr << "Computing matrix diagonals...\n";
    int rows = mat.size();
    int cols = mat[0].size();

    ptxt_mat diagonals;
    for (int i = 0; i < cols; i++)
    {
        ptxt_vec diag;
        for (int j = 0; j < rows; j++)
            diag.push_back(mat[j % rows][(i + j) % cols]);
        diagonals.push_back(diag);
    }

    return diagonals;
}

std::vector<ptxt_vec> decompose_bits(ptxt_vec values, int bitwidth)
{
    int numVals = values.size();
    std::vector<ptxt_vec> bitvecs;
    for (int i = 0; i < bitwidth; i++)
    {
        ptxt_vec bitstream;
        for (auto val : values)
            bitstream.push_back((val & (1 << i)) >> i);
        bitvecs.push_back(bitstream);
    }
    return bitvecs;
}

// turn a vector of 'long' into a single ciphertext
ctxt_vec encrypt_vector(const EncInfo &info, ptxt_vec ptxt)
{
    helib::PubKey &pk = *info.sk;
    auto ea = info.context.getEA();

    int sz = ptxt.size();
    ptxt.resize(ea.size());
    for (int i = 0; i < sz; i++) {
        ptxt[ea.size() - sz + i] = ptxt[i];
        ptxt[sz + i] = ptxt[i];
    }
    ctxt_vec zero_pad(pk);

    ea.encrypt(zero_pad, pk, ptxt);

    ctxt_vec ctxt = zero_pad;

    return ctxt;
}

ptxt_vec decrypt_vector(const EncInfo &info, ctxt_vec ctxt)
{
    ptxt_vec ptxt;
    info.context.getEA().decrypt(ctxt, *info.sk, ptxt);
    return ptxt;
}

// turn a matrix of 'long' into a vector of ciphertexts, each representing a
// generalized diagonal
ctxt_mat encrypt_matrix(const EncInfo &info, ptxt_mat ptxt)
{
    ctxt_mat mat;
    ptxt_mat diagonalized = get_diagonals(ptxt);
    for (auto vec : diagonalized) {
        mat.push_back(encrypt_vector(info, vec));
    }
        
    return mat;
}

// Encode a vector into a plaintext polynomial
zzx_vec encode_vector(const EncInfo &info, ptxt_vec vec)
{
    auto ea = info.context.getEA();
    int sz = vec.size();
    vec.resize(info.nslots);

    for (int i = 0; i < sz; i++) {
        vec[ea.size() - sz + i] = vec[i];
        vec[sz + i] = vec[i];
    }

    helib::Ptxt<helib::BGV> ptxt(info.context, vec);

    zzx_vec pp(info.context, vec);

    return pp;
}

ptxt_vec decode_vector(const EncInfo &info, zzx_vec vec)
{
    auto ea = info.context.getEA();
    ptxt_vec pp;
    ea.decode(pp, vec.getPolyRepr());
    return pp;
}

// Encode a matrix into the plaintext space
zzx_mat encode_matrix(const EncInfo &info, ptxt_mat mat)
{
    zzx_mat pp;
    for (auto diag : get_diagonals(mat)) {
        pp.push_back(encode_vector(info, diag));
    }
        
    return pp;
}

std::vector<zzx_vec> generate_rotations(zzx_vec vec, int cols, std::shared_ptr<const helib::EncryptedArray> ea)
{
    std::vector<std::vector<long>> rotations(cols, std::vector<long>(ea->size()));
    std::vector<long> raw_vec;

    ea->decode(raw_vec, vec.getPolyRepr());

    NTL_EXEC_RANGE(cols, first, last)
    {
        for (int i = first; i < last; i++)
        {
            for (int j = 0; j < cols; j++)
                rotations[j][i] = raw_vec[(i + j) % cols];
        }
    }
    NTL_EXEC_RANGE_END

    // zzx_vec encoded_rots_arr[cols];

    std::vector<zzx_vec> encoded_rotations(cols);
    NTL_EXEC_RANGE(cols, first, last)
    {
        for (int i = first; i < last; i++)
        {
            NTL::ZZX encoded;
            ea->encode(encoded, rotations[i]);
            encoded_rotations[i] = zzx_vec(ea->getContext(), encoded);
            // ea->encode(encoded_rotations[i], rotations[i]);
        }
    }
    NTL_EXEC_RANGE_END
    // std::vector<zzx_vec> encoded_rotations(std::begin(encoded_rots_arr), std::end(encoded_rots_arr));
    return encoded_rotations;
}

std::vector<ctxt_vec> generate_rotations(ctxt_vec vec, int cols, const helib::EncryptedArray& ea)
{
    std::vector<ctxt_vec> rotations(cols, ctxt_vec(vec));
    // helib::EncryptedArray ea(vec.getContext());

    ctxt_vec padded = vec;
    // ctxt_vec copy = vec;
    // ea.rotate(copy, cols);
    // padded += copy;

    int total = 0;
    NTL_EXEC_RANGE(cols, first, last)
    {
        for (int i = first; i < last; i++)
        {
            ctxt_vec rotated = padded;
            ea.rotate(rotated, -i);
            rotations[i] = rotated;
            // std::cerr << "total " << ++total << "/" << cols << "\n";
        }
    }
    NTL_EXEC_RANGE_END

    return rotations;
}

zzx_vec _innerProduct(std::vector<zzx_vec> rots, zzx_mat mat)
{
    long n = std::min(rots.size(), mat.size());
    std::vector<zzx_vec> prods(n);

    NTL_EXEC_RANGE(n, first, last)
    {
        for (int i = first; i < last; i++)
        {
            prods[i] = rots[i];
            prods[i] *= mat[i];
        }
    }
    NTL_EXEC_RANGE_END

    auto acc = prods[0];
    for (int i = 1; i < n; i++)
        acc += prods[i];

    return acc;
}

ctxt_vec _innerProduct(std::vector<ctxt_vec> rots, zzx_mat mat)
{
    long n = std::min(rots.size(), mat.size());
    std::vector<ctxt_vec> prods(n, ctxt_vec(rots[0].getPubKey()));

    NTL_EXEC_RANGE(n, first, last)
    {
        for (int i = first; i < last; i++)
        {
            prods[i] = rots[i];
            prods[i].multByConstant(mat[i]);
        }
    }
    NTL_EXEC_RANGE_END

    auto acc = prods[0];
    for (int i = 1; i < n; i++)
        acc += prods[i];

    return acc;
}

ctxt_vec _innerProduct(std::vector<ctxt_vec> rots, ctxt_mat mat)
{
    long n = std::min(rots.size(), mat.size());
    std::vector<ctxt_vec> prods(n, ctxt_vec(rots[0].getPubKey()));

    NTL_EXEC_RANGE(n, first, last)
    {
        for (int i = first; i < last; i++)
        {
            prods[i] = rots[i];
            prods[i] *= mat[i];
        }
    }
    NTL_EXEC_RANGE_END

    auto acc = prods[0];
    for (int i = 1; i < n; i++)
        acc += prods[i];

    acc.reLinearize();
    return acc;
}

ctxt_vec mat_mul(ctxt_mat mat, std::vector<ctxt_vec> rots)
{
    if (mat.size() != rots.size())
        throw std::runtime_error("Expected " + std::to_string(mat.size()) + " rotations, got " +
                                 std::to_string(rots.size()));

    return _innerProduct(rots, mat);
}

ctxt_vec mat_mul(zzx_mat mat, std::vector<ctxt_vec> rots)
{
    if (mat.size() != rots.size())
        throw std::runtime_error("Expected " + std::to_string(mat.size()) + " rotations, got " +
                                 std::to_string(rots.size()));

    return _innerProduct(rots, mat);
}

zzx_vec mat_mul(zzx_mat mat, std::vector<zzx_vec> rots)
{
    if (mat.size() != rots.size())
        throw std::runtime_error("Expected " + std::to_string(mat.size()) + " rotations, got " +
                                 std::to_string(rots.size()));

    return _innerProduct(rots, mat);
}

// ctxt_vec mat_mul(zzx_mat mat, std::vector<ctxt_vec> rots)
// {
//     if (mat.size() != rots.size())
//         throw std::runtime_error("Expected " + std::to_string(mat.size()) + " rotations, got " +
//                                  std::to_string(rots.size()));

//     return _innerProduct(rots, mat);
// }

// ctxt_vec mat_mul(ctxt_mat mat, std::vector<ctxt_vec> rots)
// {
//     if (mat.size() != rots.size())
//         throw std::runtime_error("Expected " + std::to_string(mat.size()) + " rotations, got " +
//                                  std::to_string(rots.size()));

//     return _innerProduct(rots, mat);
// }

void print_vec(std::ostream &os, ptxt_vec vec)
{
    os << "[";
    std::copy(vec.begin(), vec.end(), std::ostream_iterator<long>(os, " "));
    os << "]\n";
}