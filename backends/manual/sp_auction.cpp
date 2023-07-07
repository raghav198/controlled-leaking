#include <chrono>
#include <helib/binaryCompare.h>
#include <helib/intraSlot.h>

#include "common.hpp"

std::vector<ctxt> sp_auction_helper(std::vector<ctxt> bids, int index, ctxt top_idx, ctxt second_idx, ctxt top_bid, ctxt second_bid, EncInfo & info)
{
    std::cout << "Checking bid id " << index << "\n";
    std::vector<ctxt> result;
    if (index == bids.size()) {
        result.push_back(top_idx);
        // result.push_back(secure_index(bids, second_idx, info));
        result.push_back(second_bid);
        return result;
    }

    ctxt_bit scratch = encrypt_vector(info, ptxt());
    ctxt_bit gt(scratch), lt(scratch);

    // auto cur_best = secure_index(bids, top_idx, info);

    helib::compareTwoNumbers(gt, lt, helib::CtPtrs_vectorCt(top_bid), helib::CtPtrs_vectorCt(bids[index]));
    gt = lt;
    gt.addConstant(NTL::ZZX(1));

    auto left = sp_auction_helper(bids, index + 1, encrypt(info, ptxt{index}), top_idx, bids[index], top_bid, info);
    auto right = sp_auction_helper(bids, index + 1, top_idx, second_idx, top_bid, second_bid, info);

    // std::cout << "SP-auction with index " << index + 1 << " and top two bids " << decrypt(info, bids[index])[0] << ", " << decrypt(info, top_bid)[0] << ": " << decrypt(info, left[1])[0] << "\n";
    // std::cout << "SP-auction with index " << index + 1 << " and top two bids " << decrypt(info, top_bid)[0] << ", " << decrypt(info, second_bid)[0] << ": " << decrypt(info, right[1])[0] << "\n";
    
    result.push_back(mux(lt, gt, left[0], right[0]));
    result.push_back(mux(lt, gt, left[1], right[1]));
    return result;
}

std::vector<ctxt> sp_auction(std::vector<ctxt> bids, EncInfo & info)
{
    return sp_auction_helper(bids, 0, encrypt(info, ptxt{0}), encrypt(info, ptxt{0}), bids[0], bids[0], info);
}

int main()
{
    auto contextBuilder = helib::ContextBuilder<helib::BGV>()
        .m(8191)
        .p(2)
        .r(1)
        .bits(500)
        .c(2);
    EncInfo info(contextBuilder);
    info.context.printout();

    std::vector<long> bids = {13, 6, 11, 8, 3, 7, 17, 12};

    std::vector<ctxt> bids_c;
    ctxt lookup_c;

    for (auto val : bids)
        bids_c.push_back(encrypt(info, ptxt{val}));


    ctxt_bit scratch = encrypt_vector(info, std::vector<long>());
    ctxt_bit lt(scratch);
    ctxt_bit gt(scratch);

    ctxt zero = encrypt(info, ptxt{0});

    auto start = std::chrono::high_resolution_clock::now();
    auto output = sp_auction(bids_c, info);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long winner = decrypt(info, output[0])[0];
    long winning_bid = decrypt(info, output[1])[0];
    
    std::cout << "[ " << winner << ", " << winning_bid << " ] (" << duration.count() << "ms)\n";
    
    return 0;

}