#include "mux-common.hpp"
#include <inputs.hpp>

int main()
{
    EncInfo info(8191, 2, 1, 500, 2);

    input_handler inp;
    inp.add_arr_bits({1, 3, 2});
    inp.add_num_bits(0);

    auto data = Prep(info, inp.inputs);
    auto answer = Compute(info, data);

    auto answer_decrypted = decrypt_vector(info, answer);
    for (auto bit : answer_decrypted)
        std::cout << bit;
    std::cout << "\n";

    return 0;
}