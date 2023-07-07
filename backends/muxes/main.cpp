#include "mux-common.hpp"
#include <inputs.hpp>

int main()
{
    // EncInfo info(55831, 2, 1, 1000, 2);
    EncInfo info(8191, 2, 1, 500, 2);

    input_handler inp;
    // inp.add_arr_bits({2, 14, 4, 9, 3, 19, 15, 1}); // keys
    // inp.add_arr_bits({2, 7, 12, 8, 13, 15, 5, 15}); // values
    // inp.add_num_bits(14); // lookup
    inp.add_num_bits(4);
    inp.add_num_bits(5);
    inp.add_num_bits(3);

    auto data = Prep(info, inp.inputs);
    auto answer_enc = Compute(info, data);

    auto answer_dec = decrypt_vector(info, answer_enc);

    int answer = 0;
    int i = 1;

    for (auto lane : lanes()) {
        answer += answer_dec[lane] * i;
        i *= 2;
    }
    for (auto bit : answer_dec)
        std::cout << bit;
    std::cout << "\n";
    std::cout << "Answer: " << answer << "\n";

    return 0;
}