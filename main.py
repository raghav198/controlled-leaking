import os
from argparse import ArgumentParser

from pyparsing import ParseException

from copse_lower import generate_copse_cpp, generate_copse_data
from coyote_lower import vectorize_decisions, vectorize_labels
from holla import compile, pprint
from mux_network import codegen_mux, num, optimize_circuit, to_cpp, to_mux_network
from pita_parser import expr
from syntax_errors import report_syntax_errors


def coil_codegen(challah_tree, program_name, coil_root = 'backends/coil/coil_programs'):
    program_root = f'{coil_root}/{program_name}.coil'
    
    left_code, right_code, lt_mask, eq_mask = vectorize_decisions(challah_tree)

    label_code = vectorize_labels(challah_tree)
    masks, levels = generate_copse_data(challah_tree)
    model_code = generate_copse_cpp(masks, levels, program_name, eq_mask, lt_mask)

    os.makedirs(program_root, exist_ok=True)

    open(f'{program_root}/kernel-left.cpp', 'w').write(left_code)
    open(f'{program_root}/kernel-right.cpp', 'w').write(right_code)
    open(f'{program_root}/kernel-label.cpp', 'w').write(label_code)
    open(f'{program_root}/model.cpp', 'w').write(model_code)


def mux_network_codegen(challah_tree, program_name, mux_root = 'backends/muxes'):
    network = to_mux_network(challah_tree)
    if isinstance(network, num):
        for i, bit in enumerate(network.bits):
            print(f'[{i}] {bit}')
        print('Optimizing...')
        network.bits = [optimize_circuit(b) for b in network.bits]
        for i, bit in enumerate(network.bits):
            print(f'[{i}] {bit}')
        vector_code, vout, lanes = codegen_mux(network)
        code = to_cpp(vector_code, vout, lanes)
        
        open(f'{mux_root}/{program_name}.cpp', 'w').write(code)
        
    else:
        print('[ERROR] Array codegen is not implemented yet!')


def main(args):
    try:
        pita_program = expr.parse_file(open(args.file, encoding='utf-8'), parse_all=True)[0]
    except ParseException as parse_exception:
        report_syntax_errors(args.file)
        raise SystemExit(f'Parse error, exiting gracefully...') from parse_exception

    challah_tree = compile(pita_program)
    
    if args.show_tree:
        pprint(challah_tree)
        return
        
        
    program_name = os.path.splitext(os.path.basename(args.file))[0]

    if args.backend == 'mux':
        mux_network_codegen(challah_tree, program_name)
    else:
        coil_codegen(challah_tree, program_name)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('file')

    parser.add_argument('-b', '--backend', type=str, choices=['mux', 'coil'], default='coil')
    parser.add_argument('-s', '--show-tree', action='store_true', help='Show generated decision tree and exit')

    # parser.add_argument('-e', '--entropy', type=float, help='Maximum allowed information leakage (in bits)')
    # parser.add_argument('-r', '--num-rounds', type=int, help='Maximum number of communication rounds allowed')
    # parser.add_argument('-o', '--output', type=str, help="Where to place generated code")

    args = parser.parse_args()

    main(args)
