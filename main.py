import os
from argparse import ArgumentParser

from pyparsing import ParseException

from copse_lower import generate_copse_cpp, generate_copse_data
from coyote_lower import vectorize_decisions, vectorize_labels
from holla import compile, pprint
from mux_network import codegen_mux, num, to_mux_network
from pita_parser import expr


def coil_codegen(challah_tree, program_name):
    left_code, right_code, lt_mask, eq_mask = vectorize_decisions(challah_tree)

    label_code = vectorize_labels(challah_tree)
    masks, levels = generate_copse_data(challah_tree)
    model_code = generate_copse_cpp(masks, levels, program_name, eq_mask, lt_mask)

    os.makedirs(f'backend/{program_name}.coil/', exist_ok=True)

    open(f'coil_backend/{program_name}.coil/kernel-left.cpp', 'w').write(left_code)
    open(f'coil_backend/{program_name}.coil/kernel-right.cpp', 'w').write(right_code)
    open(f'coil_backend/{program_name}.coil/kernel-label.cpp', 'w').write(label_code)
    open(f'coil_backend/{program_name}.coil/model.cpp', 'w').write(model_code)


def mux_network_codegen(challah_tree):
    network = to_mux_network(challah_tree)
    if isinstance(network, num):
        for i, bit in enumerate(network.bits):
            print(f'[{i}] {bit}')
        vector_code = codegen_mux(network)
        for line in vector_code:
            print(f'  {line}')
    else:
        print('[ERROR] Array codegen is not implemented yet!')


def main(args):
    try:
        pita_program = expr.parse_file(open(args.file, encoding='utf-8'), parse_all=True)[0]
    except ParseException as parse_exception:
        raise SystemExit(f'Parse error: {parse_exception}') from parse_exception

    challah_tree = compile(pita_program)
    
    if args.show_tree:
        pprint(challah_tree)
        return
        
        
    program_name = os.path.splitext(os.path.basename(args.file))[0]

    if args.backend == 'mux':
        mux_network_codegen(challah_tree)
    else:
        coil_codegen(challah_tree, program_name)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('file')

    parser.add_argument('-b', '--backend', type=str, choices=['mux', 'coil'], default='coil')
    parser.add_argument('-s', '--show-tree', action='store_true', help='Show generated decision tree and quit')

    # parser.add_argument('-e', '--entropy', type=float, help='Maximum allowed information leakage (in bits)')
    # parser.add_argument('-r', '--num-rounds', type=int, help='Maximum number of communication rounds allowed')
    # parser.add_argument('-o', '--output', type=str, help="Where to place generated code")

    args = parser.parse_args()

    main(args)
