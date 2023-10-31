import os
from argparse import ArgumentParser

from pyparsing import ParseException

from copse_lower import generate_copse_cpp, generate_copse_data
from coyote_lower import vectorize_decisions, vectorize_labels
from holla import pita_compile, pprint
from lark_parse import parse_file
from pita_parser import *
from mux_network import add_depth, codegen_mux, codegen_scalar, mul_depth, num, optimize, optimize_circuit, to_cpp, to_mux_network, num_array, vec_depth
from syntax_errors import report_syntax_errors


def coil_codegen(challah_tree, program_name: str, rounds, coil_root = 'backends/coil/coil_programs'):
    program_root = f'{coil_root}/{program_name}.coil'
    
    label_code = vectorize_labels(challah_tree, rounds)
    left_code, right_code, lt_mask, eq_mask = vectorize_decisions(challah_tree, rounds)
    
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
        network.bits = [optimize(b) for b in network.bits]
        print([mul_depth(b) for b in network.bits])
        print([add_depth(b) for b in network.bits])
        # for i, bit in enumerate(network.bits):
        #     print(f'[{i}] {bit}')
        network_array = num_array(nums=[network])
    else:
        network_array = network

    vector_code, vouts, lanes, result = codegen_mux(network_array)
    print(vec_depth(vector_code))
    # input()
    
    # print('\n'.join(map(str, vector_code)))
    open(f'mux_schedules/{program_name}', 'w').write(f'lanes: {result.lanes}\nalignment: {result.alignment}')
    code = to_cpp(vector_code, vouts, lanes, result)
    
    open(f'{mux_root}/{program_name}.cpp', 'w').write(code)
        

def scalar_codegen(challah_tree, program_name, scalar_root = 'backends/scalar'):
    network = to_mux_network(challah_tree)
    if isinstance(network, num):
        # for i, bit in enumerate(network.bits):
        #     print(f'[{i}] {bit}')
        network.bits = [optimize(b) for b in network.bits]
        print([mul_depth(b) for b in network.bits])
        
        for i, bit in enumerate(network.bits):
            print(f'[{i}] {bit}')
        network_array = num_array(nums=[network])
    else:
        network_array = num_array(nums=[num(bits=list(map(optimize, bit.bits))) for bit in network.nums])
 
    code = codegen_scalar(network_array)
    open(f'{scalar_root}/{program_name}.cpp', 'w').write(code)
    
    
def cached_codegen(cache_path: str):
    pass
    


def main(args):
    try:
        pita_program = expr.parse_file(open(args.file, encoding='utf-8'), parse_all=True)[0]
        # pita_program = parse_file(open(args.file, encoding='utf-8'))
    except ParseException as parse_exception:
        report_syntax_errors(args.file)
        raise SystemExit('Parse error, exiting gracefully...') from parse_exception

    challah_tree = pita_compile(pita_program)
    
    if args.show_tree:
        pprint(challah_tree)
        return
        
        
    program_name = os.path.splitext(os.path.basename(args.file))[0]

    if args.backend == 'mux':
        mux_network_codegen(challah_tree, program_name)
    elif args.backend == 'scalar':
        scalar_codegen(challah_tree, program_name)
    else:
        coil_codegen(challah_tree, program_name, args.rounds)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('file')

    parser.add_argument('-b', '--backend', type=str, choices=['mux', 'coil', 'scalar'], default='coil')
    parser.add_argument('-s', '--show-tree', action='store_true', help='Show generated decision tree and exit')
    
    parser.add_argument('--rounds', type=int, default=10)

    # parser.add_argument('-e', '--entropy', type=float, help='Maximum allowed information leakage (in bits)')
    # parser.add_argument('-r', '--num-rounds', type=int, help='Maximum number of communication rounds allowed')
    # parser.add_argument('-o', '--output', type=str, help="Where to place generated code")

    args = parser.parse_args()

    main(args)
