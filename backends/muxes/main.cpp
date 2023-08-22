#include "mux-common.hpp"
#include <inputs.hpp>

int main()
{
    // EncInfo info(55831, 2, 1, 1000, 2);
    EncInfo info(8191, 2, 1, 1000, 2);

    input_handler inp;
    // linear_oram, log_oram
    inp.add_arr_bits({9, 2, 3, 12, 6, 8, 7, 1, 4, 5, 0, 10, 21, 16, 30, 13}); // array
    inp.add_num_bits(6); // index

    // // filter
    // inp.add_arr_bits({14, 15, 9, 13, 6, 16, 19, 10}); // array
    // inp.add_num_bits(11); // threshold

    // // merge
    // inp.add_arr_bits({7, 12, 17, 18, 20}); // array 1
    // inp.add_arr_bits({3, 5, 11, 14, 19}); // array 2

    // sp_auction
    // inp.add_arr_bits({13, 6, 11, 8, 3, 7, 17, 12}); // bids

    // // associative_array
    // inp.add_arr_bits({2, 14, 4, 9, 3, 19, 15, 1}); // keys
    // inp.add_arr_bits({2, 7, 12, 8, 13, 15, 5, 15}); // values
    // inp.add_num_bits(4); // lookup

    auto data = Prep(info, inp.inputs);
    auto start = std::chrono::high_resolution_clock::now();
    auto answer_enc = Compute(info, data);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::vector<ptxt_vec> answer_dec;
    for (auto out : answer_enc) answer_dec.push_back(decrypt_vector(info, out));

    // auto answer_dec = decrypt_vector(info, answer_enc);

    std::vector<int> answers;

    auto L = lanes();
    for (int out_index = 0; out_index < L.size(); out_index++) {
        int answer = 0;
        int i = 1;
        for (auto lane : L[out_index]) {
            answer += answer_dec[out_index][lane] * i;
            i *= 2;
        }
        answers.push_back(answer);
    }

    // for (auto out_lanes : lanes()) {
    //     int answer = 0;
    //     int i = 1;
    //     for (auto lane : out_lanes){
    //         answer += answer_dec[lane] * i;
    //         i *= 2;
    //     }
    //     answers.push_back(answer);
    // }
    // for (auto bit : answer_dec)
    //     std::cout << bit;
    std::cout << "\n";
    std::cout << "Answer: ";
    for (auto answer : answers)
        std::cout << answer << " ";
    std::cout << " (" << duration.count() << "ms)\n";

    return 0;
}