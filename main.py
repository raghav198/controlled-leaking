from argparse import ArgumentParser
import os
from coyote_lower import vectorize_decisions, vectorize_labels
from pita_parser import expr
from holla import compile, pprint
from pyparsing import ParseException
from slice_program import get_interactive_layers, basic_pprint
from copse_lower import generate_copse_cpp, generate_copse_data

def show_layers(challah_tree):
    challah_layers, basic_layers = get_interactive_layers(challah_tree, args.num_rounds, args.entropy)
    
    print('Basic layers:')
    for layer in basic_layers[::-1]:
        for tree in layer:
            basic_pprint(tree)
            print('-' * 10)
        input('=' * 10)
    
    print('Challah layers:')
    for layer in challah_layers[::-1]:
        for tree in layer:
            pprint(tree)
            print('-' * 10)
        input('=' * 10)
        

def show_copse_data(challah_tree):
    masks, matrices = generate_copse_data(challah_tree)


    for i, (mask, matrix) in enumerate(zip(masks, matrices)):
        print(f'-- Level {i + 1} --')
        print(f'Mask: {mask}')
        print('Matrix:')
        for row in matrix:
            print(row)


def show_coyote_circuits(challah_tree):
    pprint(challah_tree)
    vectorize_decisions(challah_tree)
    
    
includes = """
#include "kernel-left.hpp"
#include "kernel-right.hpp"
#include "kernel-label.hpp"
#include "model.hpp"
"""

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('file')
    
    parser.add_argument('-e', '--entropy', type=float, help='Maximum allowed information leakage (in bits)')
    parser.add_argument('-r', '--num-rounds', type=int, help='Maximum number of communication rounds allowed')
    parser.add_argument('-o', '--output', type=str, help="Where to place generated code")
    
    args = parser.parse_args()
    
    try:
        pita_program = expr.parse_file(open(args.file), parse_all=True)[0]
    except ParseException as e:
        raise SystemExit(f'Parse error: {e}')
    
    challah_tree = compile(pita_program)
    pprint(challah_tree)
    program_name = os.path.splitext(os.path.basename(args.file))[0]
    
    left_code, right_code = vectorize_decisions(challah_tree)

    label_code = vectorize_labels(challah_tree)
    masks, levels = generate_copse_data(challah_tree)
    model_code = generate_copse_cpp(masks, levels, program_name)
    
    os.makedirs(f'backend/{program_name}.coil/', exist_ok=True)
    
    open(f'backend/{program_name}.coil/kernel-left.hpp', 'w').write(left_code)
    open(f'backend/{program_name}.coil/kernel-right.hpp', 'w').write(right_code)
    open(f'backend/{program_name}.coil/kernel-label.hpp', 'w').write(label_code)
    open(f'backend/{program_name}.coil/model.hpp', 'w').write(model_code)
    open(f'backend/{program_name}.coil/{program_name}.hpp', 'w').write(includes)
    