#pragma once

#include <initializer_list>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>

struct input_handler {
    int num_inputs = 0;
    std::unordered_map<std::string, int> inputs;
    
    void add_num(int val) {
        std::ostringstream name;
        name << "input#" << num_inputs++;

        inputs[name.str()] = val;
    }
    
    void add_num_bits(int val, int max_bits = 8) {
        for (int i = 0; i < max_bits; i++) {
            std::ostringstream name;
            name << "input#" << num_inputs << "." << i;
            inputs[name.str()] = ((1 << i) & val) >> i;
        }
        num_inputs++;
    }

    void add_arr(std::initializer_list<int> vals) {
        for (auto val : vals) {
            add_num(val);
        }
    }

    void add_arr_bits(std::initializer_list<int> vals, int max_bits = 8) {
        for (auto val : vals) {
            add_num_bits(val, max_bits);
        }
    }

    void add_from_file(char *file) {
        std::ifstream fin(file);
        if (fin.is_open()) {
            int val;
            while (fin >> val) {
                add_num(val);
            }
        } else {
          std::cout << "Cannot read input file\n";
          exit(1);
        }
    }
};
