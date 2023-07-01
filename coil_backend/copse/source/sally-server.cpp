#include "sally-server.hpp"

#include <chrono>
#include <functional>

/**
 * The only time the feature vectors are unencrypted is if the model is also plaintext
 * (Otherwise it doesn't make any sense, because they have to immediately be upcast anyway)
 */

#define cmp_vec data_vec

void mult(ctxt_vec &v1, ctxt_vec v2)
{
    v1.multiplyBy(v2);
}

void mult(ctxt_vec &v1, zzx_vec v2)
{
    v1.multByConstant(v2);
}

void mult(zzx_vec &v1, zzx_vec v2)
{
    v1 *= v2;
}

void add(ctxt_vec &v1, ctxt_vec v2)
{
    v1 += v2;
}

void add(ctxt_vec &v1, zzx_vec v2)
{
    v1.addConstant(v2);
}

void add(zzx_vec &v1, zzx_vec v2)
{
    v1 += v2;
}

std::vector<cmp_vec> prefix_mult(std::vector<cmp_vec> vals)
{
    int N = vals.size();
    std::vector<cmp_vec> acc = vals, cur = vals;
    std::vector<int> depths(acc.size());

    for (int i = 1; i < N; i <<= 1)
    {
        acc = cur;
        NTL_EXEC_INDEX(N - i, j)
        {
            cur[j] = acc[j];
            mult(cur[j], acc[j + 1]);
            // cur[j].multiplyBy(acc[j + i]);
            depths[j]++;
        }
        NTL_EXEC_INDEX_END
    }

    return cur;
}

cmp_vec reduce(std::vector<cmp_vec> vals, std::function<cmp_vec(cmp_vec, cmp_vec)> func, int start, int end)
{

    using clock = std::chrono::high_resolution_clock;
    using ms = std::chrono::milliseconds;

    if (end - start == 0 || vals.size() == 0)
        throw new std::runtime_error("Cannot reduce size-0 vector");
    if (end - start == 1)
        return vals[start];

    int mid = start + (end - start) / 2;

    auto lhs = reduce(vals, func, start, mid);
    auto rhs = reduce(vals, func, mid, end);

    auto t1 = clock::now();
    auto answer = func(lhs, rhs);
    auto t2 = clock::now();

    std::cout << "func call time for (" << start << ", " << end
              << ") = " << std::chrono::duration_cast<ms>(t2 - t1).count() << "ms\n";

    return func(lhs, rhs);
}

cmp_vec compare(std::vector<model_vec> thresholds, std::vector<data_vec> features)
{

    using clock = std::chrono::high_resolution_clock;
    using ms = std::chrono::milliseconds;

    // auto pk = features[0].getPubKey();
    int N = thresholds.size();
    std::vector<cmp_vec> L, Q;

    auto t1 = clock::now();
    for (int i = 0; i < N; i++)
    {
        data_vec _l = features[i];
        data_vec _q = features[i];

        _l.addConstant(NTL::ZZX(1));
        // add(_l, NTL::ZZX(1));
        mult(_l, thresholds[i]);
        // _l.multiplyBy(thresholds[i]);

        _q.addConstant(NTL::ZZX(1));
        // add(_q, NTL::ZZX(1));
        add(_q, thresholds[i]);
        // _q += thresholds[i];

        L.push_back(_l);
        Q.push_back(_q);
    }
    auto t2 = clock::now();

    std::vector<cmp_vec> E = prefix_mult(Q);
    auto t3 = clock::now();
    NTL_EXEC_INDEX(N - 1, i)
    mult(E[i + 1], L[i]);
    // E[i + 1].multiplyBy(L[i]);
    NTL_EXEC_INDEX_END
    auto t4 = clock::now();

    cmp_vec result = reduce(
        E,
        [](cmp_vec lhs, cmp_vec rhs) {
            cmp_vec prod = lhs;
            mult(prod, rhs);
            // prod.multiplyBy(rhs);

            add(lhs, rhs);
            add(lhs, prod);
            // lhs += rhs;
            // lhs += prod;

            return lhs;
        },
        0, E.size());

    auto t5 = clock::now();

    cmp_vec prod = result;
    mult(prod, L[N - 1]);
    // prod.multiplyBy(L[N - 1]);
    add(result, L[N - 1]);
    add(result, prod);
    // result += L[N - 1];
    // result += prod;
    auto t6 = clock::now();

    std::cout << "Timings: " << std::chrono::duration_cast<ms>(t2 - t1).count() << "; "
              << std::chrono::duration_cast<ms>(t3 - t2).count() << "; "
              << std::chrono::duration_cast<ms>(t4 - t3).count() << "; "
              << std::chrono::duration_cast<ms>(t5 - t4).count() << "; "
              << std::chrono::duration_cast<ms>(t6 - t5).count() << "; ";

    return result;
}

void SallyServer::LoadModel()
{
    if (model != nullptr)
        delete model;
    model = maurice->GenerateModel();
}

void SallyServer::ExecuteQuery()
{

    using ms = std::chrono::milliseconds;
    using clock = std::chrono::high_resolution_clock;

    if (model == nullptr)
        return;

    std::vector<std::chrono::milliseconds> times;

    auto features = danielle->GetFeatureVector(model->k, model->thresholds.size());

    Timer<ms> timer(times);
    // auto t1 = clock::now();
    auto decisions = compare(model->thresholds, features.bitwise_features);
    // auto t2 = clock::now();
    timer.measure();
    // times.push_back(std::chrono::duration_cast<ms>(t2 - t1));

    // t1 = clock::now();
    auto decision_rots = generate_rotations(decisions, model->d2b.size(), ea);
    auto branches = model->d2b.size() == 0 ? decisions : mat_mul(model->d2b, decision_rots);
    // t2 = clock::now();
    // times.push_back(std::chrono::duration_cast<ms>(t2 - t1));
    timer.measure();

    int num_levels = model->level_b2s.size();

    std::vector<cmp_vec> masks(num_levels, branches);

    // t1 = clock::now();
    auto branch_rots = generate_rotations(branches, model->level_b2s[0].size(), ea);
    for (int i = 0; i < num_levels; i++)
    {
        auto b2s = model->level_b2s[i];
        auto mask = model->level_mask[i];

        auto slots = mat_mul(b2s, branch_rots);
        add(slots, mask);
        // mask += slots;

        masks[i] = slots;
    }
    timer.measure();
    // t2 = clock::now();
    // times.push_back(std::chrono::duration_cast<ms>(t2 - t1));

    // t1 = clock::now();
    auto inference = reduce(
        masks,
        [](auto lhs, auto rhs) {
            mult(lhs, rhs);
            // lhs.multiplyBy(rhs);
            return lhs;
        },
        0, masks.size());
    // t2 = clock::now();
    timer.measure();
    // times.push_back(std::chrono::duration_cast<ms>(t2 - t1));

    std::cout << model->name << "," << times[0].count() << "," << times[1].count() << "," << times[2].count() << ","
              << times[3].count() << "\n";
    danielle->accept_inference(inference);
}
