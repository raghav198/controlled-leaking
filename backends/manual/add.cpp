#include "common.hpp"

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

    std::cout << bitwidth << "\n";

    ctxt a = encrypt(info, ptxt{4});
    ctxt b = encrypt(info, ptxt{5});
    ctxt c = encrypt(info, ptxt{3});
    auto start = std::chrono::high_resolution_clock::now();
    ctxt answer = mul(a, add(b, c));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << decrypt(info, answer)[0] << "(" << duration.count() << "ms)\n";
    
    return 0;

}