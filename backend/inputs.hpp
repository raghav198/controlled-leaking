#pragma once

#include <initializer_list>
#include <sstream>
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
    void add_arr(std::initializer_list<int> vals) {
        for (auto val : vals) {
            add_num(val);
        }
    }
};