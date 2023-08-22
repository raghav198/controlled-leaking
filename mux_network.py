from collections import defaultdict
from dataclasses import dataclass
from math import ceil, log
from typing import Callable, ClassVar, cast

from coyote import coyote_ast
from coyote.codegen import (Schedule, VecBlendInstr, VecConstInstr, VecInstr,
                            VecLoadInstr, VecOpInstr, VecRotInstr)
from coyote.coyote_ast import CompilerV2
from coyote.vectorize_circuit import CodeObject, vectorize, vectorize_cached

from coyote_lower import analyzeV2
from holla import (ChallahArithExpr, ChallahArray, ChallahBranch, ChallahLeaf,
                   ChallahTree, ChallahVar)

bit = coyote_ast.Expression


@dataclass
class num:
    """Bits of a number represented least significant bit first (e.g. `12` as a four bit number is `0011`)"""
    
    bitwidth: ClassVar[int] = 8
    
    bits: list[bit]

    def __post_init__(self):
        assert len(self.bits) <= num.bitwidth
        while len(self.bits) < num.bitwidth:
            self.bits.append(coyote_ast.Var('0'))


@dataclass
class num_array:
    nums: list[num]


def balanced_reduce(values: list[bit], function: Callable[[bit, bit], bit]) -> bit:
    if len(values) == 0:
        raise RuntimeError('reducing empty list!')
    if len(values) == 1:
        return values[0]
    mid = len(values) // 2
    left_reduction = balanced_reduce(values[:mid], function)
    right_reduction = balanced_reduce(values[mid:], function)
    return function(left_reduction, right_reduction)


def eq_circuit(left: num, right: num) -> bit:
    xors: list[bit] = [left_bit + right_bit + coyote_ast.Var('1') for left_bit, right_bit in zip(left.bits, right.bits)]
    return balanced_reduce(xors, lambda a, b: a * b)


def lt_circuit(left: num, right: num) -> bit:
    eq_bits = [left_bit + right_bit + coyote_ast.Var('1') for left_bit, right_bit in zip(left.bits, right.bits)]
    lt_bits = [right_bit * (left_bit + coyote_ast.Var('1')) for left_bit, right_bit in zip(left.bits, right.bits)]

    running_eq = [coyote_ast.Var('1'), eq_bits[-1]]  # running_eq[i]: are the numbers equal *up to bit `i`*?
    for bit in eq_bits[:-1][::-1]:
        running_eq.append(running_eq[-1] * bit)
    running_eq = running_eq[::-1]

    # OR the partial comparisons together!
    return balanced_reduce([eq_bit * lt_bit for eq_bit, lt_bit in zip(running_eq[1:], lt_bits)], lambda a, b: a * b + a + b)


def mux(condition: bit, true: num, false: num) -> num:
    def mux_bit(condition: bit, true: bit, false: bit): 
        return condition * true + (coyote_ast.Var('1') + condition) * false
    return num([mux_bit(condition, true_bit, false_bit) for true_bit, false_bit, in zip(true.bits, false.bits)])


def mux_arrays(condition: bit, trues: num_array, falses: num_array) -> num_array:
    return num_array([mux(condition, true, false) for true, false in zip(trues.nums, falses.nums)])


def to_bits(val: int, twos_complement=True) -> num:
    if twos_complement and val < 0:
        val += 2 ** (num.bitwidth - 1)
    assert val < 2 ** num.bitwidth, f'Value {val} too big for {num.bitwidth} bits'
    return num([coyote_ast.Var(b) for b in bin(val)[2:][::-1]])


def negate(val: num) -> num:
    flipped_bits = [val.bits[0]] + [b + coyote_ast.Var('1') for b in val.bits[1:]]
    carry = val.bits[0] + coyote_ast.Var('1')
    for i in range(len(flipped_bits) - 1):
        flipped_bits[i + 1], carry = flipped_bits[i + 1] + carry, flipped_bits[i + 1] * carry
    return num(flipped_bits)


def add(left: num, right: num) -> num:
    sum_bits: list[bit] = [left.bits[0] + right.bits[0]]
    carry = left.bits[0] * right.bits[0]
    for left_bit, right_bit in zip(left.bits[1:], right.bits[1:]):
        sum_bits.append(left_bit + right_bit + carry)
        carry = left_bit * right_bit + carry * (left_bit + right_bit)
    return num(sum_bits)


def mul(left: num, right: num) -> num:
    product_bits: list[bit] = []

    for k in range(num.bitwidth):
        accumulate = []
        for i in range(k + 1):
            accumulate.append(left.bits[i] * right.bits[k - i])
        product_bits.append(balanced_reduce(accumulate, lambda a, b: a + b))

    return num(product_bits)


def sub(left: num, right: num) -> num:
    return add(left, negate(right))


def num_circuit(expr: ChallahLeaf) -> num:
    match expr:
        case ChallahVar(name):
            if name.isnumeric():
                return to_bits(int(name))
            return num([coyote_ast.Var(f'{name}.{b}') for b in range(num.bitwidth)])
        case ChallahArithExpr(left, op, right):
            if op == '+':
                return add(num_circuit(left), num_circuit(right))
            if op == '-':
                return sub(num_circuit(left), num_circuit(right))
            if op == '*':
                return mul(num_circuit(left), num_circuit(right))
            raise TypeError(f'Unrecognized operator `{op}`')
    raise TypeError(type(expr))


def interpret_bit(circuit: bit, inputs: dict[str, int] = {}) -> int:
    match circuit:
        case coyote_ast.Var(name):
            if name.isnumeric():
                return int(name)
            base, bit_num = name.split('.')
            return int(inputs[base] & (2 ** int(bit_num)) == 0)
        case coyote_ast.Op(op, lhs, rhs):
            if op not in '+*':
                raise TypeError(f'`{op}` is not a boolean operator')
            if op == '+':
                return interpret_bit(lhs, inputs) ^ interpret_bit(rhs, inputs)
            return interpret_bit(lhs, inputs) & interpret_bit(rhs, inputs)
    raise TypeError(type(circuit))


def interpret(circuit: num, inputs: dict[str, int] = {}) -> int:
    bits = [interpret_bit(b, inputs) for b in circuit.bits]
    return sum(b * 2 ** i for i, b in enumerate(bits))


def to_mux_network(tree: ChallahTree) -> num | num_array:
    match tree:
        case ChallahBranch(left, lt, right, true, false):
            condition = (lt_circuit if lt else eq_circuit)(num_circuit(left), num_circuit(right))
            true_out = to_mux_network(true)
            false_out = to_mux_network(false)
            assert isinstance(true_out, num_array) == isinstance(false_out, num_array)
            if isinstance(true_out, num_array) and isinstance(false_out, num_array):
                return mux_arrays(condition, true_out, false_out)
            assert isinstance(true_out, num) and isinstance(false_out, num)
            return mux(condition, true_out, false_out)
        case ChallahArray(elems):
            return num_array([num_circuit(elem) for elem in elems])
        case _:
            if isinstance(tree, ChallahLeaf):
                return num_circuit(tree)
            raise TypeError(type(tree))


def to_cpp(code: list[VecInstr], vouts: list[int], output_lanes: list[list[int]], code_object: CodeObject):

    def reg2var(reg: str) -> str:
        if reg in aliases:
            return aliases[reg]
        return reg[2:]

    aliases: dict[str, str] = {}
    compute_lines: list[str] = []
    prep_lines: list[str] = []
    all_masks: set[str] = set()
    
    # prune things we know are all zero
    all_zero: set[str] = set()
    zero_constant = ''
    for instr in code:
        if isinstance(instr, VecConstInstr) and all(val == '0' for val in instr.vals):
            all_zero.add(instr.dest)
            if not zero_constant:
                zero_constant = instr.dest
        if isinstance(instr, VecOpInstr) and instr.op == '*' and (instr.lhs in all_zero or instr.rhs in all_zero):
            all_zero.add(instr.dest)
            aliases[instr.dest] = zero_constant
        if isinstance(instr, VecRotInstr) and instr.operand in all_zero:
            all_zero.add(instr.dest)
            aliases[instr.dest] = zero_constant
        if isinstance(instr, VecBlendInstr):
            new_parts = [(val, mask) for val, mask in zip(instr.vals, instr.masks) if val not in all_zero]
            instr.vals, instr.masks = zip(*new_parts)
            
    code = [instr for instr in code if instr.dest not in all_zero]
    
    # first do some fucking register allocation
    last_use: dict[str, int] = defaultdict(lambda: len(code))
    for j, instr in enumerate(reversed(code)):
        i = len(code) - j
        match instr:
            case VecBlendInstr(dest, vals, mask):
                last_use.update({val: i for val in set(vals) - set(last_use)})
                # for val in set(vals) - set(last_use):
                    # print(f'last use of {val} = {i}')
            case VecLoadInstr(dest, src):
                last_use[src] = last_use[dest]
            case VecOpInstr(dest, op, lhs, rhs):
                last_use.setdefault(lhs, i)
                last_use.setdefault(rhs, i)
                # print(f'last use of {lhs} and {rhs} = {i}')
                # last_use.update({lhs: i, rhs: i})
            case VecRotInstr(dest, operand, shift):
                last_use.setdefault(operand, i)
                # print(f'Last use of {operand} = {i}')
                # if operand not in last_use:
                #     last_use[operand] = i
    
    for instr in code:
        if isinstance(instr, VecConstInstr):
            last_use[instr.dest] = len(code) # this is a hack
                
    # update `aliases` with allocated registers
    allocation: dict[str, str] = {}
    def same_type(reg: str):
        return list(filter(lambda name: name[0] == reg[2], allocation.keys()))
    
    for i, line in enumerate(code):
        if isinstance(line, VecConstInstr):
            continue
        
        # print(f'Allocating for {line.dest}:', end=' ')
        available = same_type(line.dest)
        usable = list(filter(lambda name: last_use[allocation[name]] < i, available))
        if usable:
            # print(f'Kicking out {allocation[usable[0]]} on {usable[0]}; it died on {last_use[allocation[usable[0]]]} (currently {i})')
            new_name = usable[0]
        else:
            new_name = f'{line.dest[2]}[{len(available)}]'
            # print(f'Generating new register: {new_name}')
        
        allocation[new_name] = line.dest
        aliases[line.dest] = new_name
        
        
    declared: set[str] = set() # set of variables declared so far
    def assign(name, value):
        ret = f'{name} = {value};' if name in declared else f'{name[0]}.push_back({value});'
        declared.add(name)
        return ret
        
        
    num_ptxts: int = 0
    num_ctxts: int = 0
    
    rotations: dict[str, int] = defaultdict(int)
    
    for line in code:
        match line:
            case VecRotInstr(dest, operand, shift):
                rotations[reg2var(dest)] = rotations[reg2var(operand)] + shift
                
                compute_lines += [
                    assign(reg2var(dest), reg2var(operand)),
                    # f'{decltype(reg2var(dest))} = {reg2var(operand)};',
                    f'info.context.getEA().rotate({reg2var(dest)}, {shift});',
                ]
                
            case VecOpInstr(dest, op, lhs, rhs):
                rotations[reg2var(dest)] = max(rotations[reg2var(lhs)], rotations[reg2var(rhs)])
                
                if op == '-':
                    op = '+'
                    print('[WARN] No subtraction in mod 2 >:(')
                
                if reg2var(lhs).startswith('data.plaintexts') and reg2var(rhs).startswith('data.plaintexts'):
                    compute_lines += [
                        f'data.plaintexts.push_back({reg2var(lhs)});',
                        f'data.plaintexts[{num_ptxts}] {op}= {reg2var(rhs)};'
                    ]
                    aliases[dest] = f'data.plaintexts["{num_ptxts}"]'
                    num_ptxts += 1
                    continue
                
                if reg2var(lhs).startswith('data.plaintexts') and not reg2var(rhs).startswith('data.plaintexts'):
                    lhs, rhs = rhs, lhs
                
                compute_lines += [
                    assign(reg2var(dest), reg2var(lhs)),
                    # f'{decltype(reg2var(dest))} = {reg2var(lhs)};',
                    f'{reg2var(dest)}.multiplyBy({reg2var(rhs)});' if op == '*' and not reg2var(rhs).startswith('data.plaintexts') else f'{reg2var(dest)} {op}= {reg2var(rhs)};'
                ]
                
            case VecLoadInstr(dest, src):
                aliases[dest] = reg2var(src)
            case VecConstInstr(dest, vals):
                ptxt_vec = [f'inputs["{val}"]' if not val.isnumeric() else val for val in vals]
                to_encrypt = any(not val.isnumeric() for val in vals)
                prep_dest = 'data.ciphertexts' if to_encrypt else 'data.plaintexts'
                encoder = 'encrypt_vector' if to_encrypt else 'encode_vector'
                prep_lines += [
                    f'ptxt {reg2var(dest).replace("[", "").replace("]", "")}{{{", ".join(ptxt_vec)}}};',
                    f'{prep_dest}.push_back({encoder}(info, {reg2var(dest).replace("[", "").replace("]", "")}, pad_positive, pad_negative));'
                ]
                
                if prep_dest == 'data.ciphertexts':
                    num_ctxts, dest_idx = num_ctxts + 1, num_ctxts
                else:
                    num_ptxts, dest_idx = num_ptxts + 1, num_ptxts
                
                aliases[dest] = f'{prep_dest}[{dest_idx}]'
            case VecBlendInstr(dest, vals, masks):
                rotations[reg2var(dest)] = max(rotations[reg2var(val)] for val in vals)
                
                all_masks |= {'"' + ''.join(map(str, mask)) + '"' for mask in masks}
                pairs = [(v, m) for v, m in zip(vals, masks) if not reg2var(v).startswith('data.plaintexts')]
                args = [f'{{{reg2var(val)}, data.masks["{"".join(map(str, mask))}"]}}' for val, mask in pairs]
                compute_lines += [
                    assign(reg2var(dest), f'blend_bits({{{", ".join(args)}}})')
                    # f'{decltype(reg2var(dest))} = blend_bits({{{", ".join(args)}}});'
                ]
                
                ptxt_pairs = [(v, m) for v, m in zip(vals, masks) if reg2var(v).startswith('data.plaintexts')]
                if ptxt_pairs:
                    compute_lines += ['{']
                    for i, (v, m) in enumerate(ptxt_pairs):
                        compute_lines += [
                            f'    auto b{i} = {reg2var(v)};'
                            f'    b{i} *= data.masks["{"".join(map(str, m))}"];',
                            f'    {reg2var(dest)} += b{i};']
                    compute_lines += ['}']
                
            case _:
                raise TypeError(line)
        
        if compute_lines and compute_lines[-1].startswith(('v', 'info.context')):
            compute_lines.append(f'show({reg2var(dest)});')

    prep_lines += [f'data.masks[{mask}] = make_mask(info, {mask}, pad_positive, pad_negative);' for mask in all_masks]
    
    prep_lines.append('return data;')
    # print(f'aliases: {aliases}')
    values = ", ".join([reg2var(f'__v{vout}') for vout in vouts])
    # print(f'output values: {values}')
    # print(f'(vouts: {vouts})')
    
    parts = [compute_lines[i:i+1000] for i in range(0, len(compute_lines), 1000)]
    part_template = """
void part{n}(EncInfo &info, compute_data data, std::vector<ctxt_bit> &v, std::vector<ctxt_bit> &s, std::vector<ctxt_bit> &t)
{{
    {code}
}}
"""

    parts_code = '\n\n'.join([part_template.format(n=i, code='\n    '.join(part)) for i, part in enumerate(parts)])
    
    # compute_lines.append(f'return {reg2var(f"__v{vout}")};')
    
    output_lane_vectors = [f"{{{', '.join(map(str, arr_output_lanes))}}}" for arr_output_lanes in output_lanes]
    
    prep = "\n    ".join(prep_lines)
    compute = '\n    '.join([f'part{i}(info, data, v, s, t);' for i in range(len(parts))] + [f'return std::vector<ctxt_bit>{{ {values} }};'])
    # compute = "\n    ".join(compute_lines)
    
    width = len(next(iter(all_masks)))
    # print(next(iter(all_masks)))
    
    print(f'rotations: {rotations}')
    
    return f"""
#include <helib/FHE.h>

#include "mux-common.hpp"

#ifdef DEBUG
#define show(x) _show(info, x, #x, {width - 2})
#else
#define show(x)
#endif

void _show(EncInfo & info, ctxt_bit vec, std::string name, int size)
{{
    std::cout << name << ": " << name.capacity() << "\\n";
    /*bool left_ok = true;
    bool right_ok = true;
    auto decrypted = decrypt_vector(info, vec);
    for (int i = 0; i < size; i++) {{
        std::cout << decrypted[i];
        if (decrypted[i] != decrypted[i + size]) right_ok = false;
        if (decrypted[i] != decrypted[decrypted.size() - size + i]) left_ok = false;
    }}

    if (right_ok || left_ok) std::cout << "\\t*";
    else std::cout << "\\tX";

    std::cout << "\\n";*/
}}

std::vector<std::vector<int>> lanes()
{{
    return std::vector<std::vector<int>>{{{", ".join(map(str, output_lane_vectors))}}};
}}

compute_data Prep(EncInfo & info, std::unordered_map<std::string, int> inputs)
{{

    compute_data data;
    int pad_positive = {ceil(code_object.max_rotation / width)};
    int pad_negative = {-ceil(code_object.min_rotation / width)};
    
    pad_positive = 1;
    pad_negative = 1;
    
    {prep}
}}

{parts_code}

std::vector<ctxt_bit> Compute(EncInfo & info, compute_data data)
{{
    std::vector<ctxt_bit> v, s, t;
    {compute}
}}
    """

def fixpoint(f):
    def inner(c, debug=False, maxdepth=3):
        new_c = None
        for _ in range(maxdepth):
            new_c = f(c, debug)
            if c == new_c:
                break
            c = new_c

        return new_c
        
        
    return inner


def id_cache(f: Callable[[bit], bit]):
    cache: dict[int, bit] = {}
    def inner(c: bit, debug=False) -> bit:
        if id(c) in cache:
            return cache[id(c)]
        value = f(c, debug)
        cache[id(c)] = value
        return value
    return inner


def logging(f: Callable[[bit], bit]):
    def inner(c: bit, debug=False) -> bit:
        value = f(c, debug)
        if value != c:
            print(f'Optimized {c} to {value}')
        return value
    return inner

def optimize(circuit: bit) -> bit:
    def clean(gate: coyote_ast.Op) -> bit:
        match gate:
            case coyote_ast.Op('+', coyote_ast.Var('0'), expr) | coyote_ast.Op('+', expr, coyote_ast.Var('0')):
                return expr
            case coyote_ast.Op('*', coyote_ast.Var('0'), expr) | coyote_ast.Op('*', expr, coyote_ast.Var('0')):
                return coyote_ast.Var('0')
            case coyote_ast.Op('*', coyote_ast.Var('1'), expr) | coyote_ast.Op('*', expr, coyote_ast.Var('1')):
                return expr
            case _:
                return gate
            
    if isinstance(circuit, coyote_ast.Var):
        return circuit
    assert isinstance(circuit, coyote_ast.Op)
    circuit.lhs = optimize(circuit.lhs)
    circuit.rhs = optimize(circuit.rhs)
    return clean(circuit)
    

def mul_depth(circuit: bit):
    match circuit:
        case coyote_ast.Var(_):
            return 0
        case coyote_ast.Op(op, lhs, rhs):
            if op == '*':
                return 1 + max(mul_depth(lhs), mul_depth(rhs))
            return max(mul_depth(lhs), mul_depth(rhs))

def add_depth(circuit: bit):
    match circuit:
        case coyote_ast.Var(_):
            return 0
        case coyote_ast.Op(op, lhs, rhs):
            if op == '+':
                return 1 + max(mul_depth(lhs), mul_depth(rhs))
            return max(mul_depth(lhs), mul_depth(rhs))
        

def vec_depth(code: list[VecInstr]):
    depths: dict[str, tuple[int, int]] = {}
    for line in code:
        match line:
            case VecOpInstr(dest, op, lhs, rhs):
                depths[dest] = (max(depths[lhs][0], depths[rhs][0]) + (op == '+'),
                                max(depths[lhs][1], depths[rhs][1]) + (op == '*'))
            case VecRotInstr(dest, operand, shift):
                depths[dest] = depths[operand]
            case VecLoadInstr(dest, src):
                depths[dest] = depths[src]
            case VecConstInstr(dest, vals):
                depths[dest] = (0, 0)
            case VecBlendInstr(dest, vals, masks):
                adds = max(depths[val][0] for val in vals)
                muls = max(depths[val][1] for val in vals)
                # adds += ceil(log(len(vals)) / log(2))
                depths[dest] = (adds, muls)
    return depths
                

# @logging
@id_cache
@fixpoint
def optimize_circuit(circuit: bit, debug=False) -> bit:
    match circuit:
        case coyote_ast.Var(_):
            return circuit
        case coyote_ast.Op(op, lhs, rhs):
            match op, lhs, rhs:
                case ('+', coyote_ast.Var('0'), expr) | ('+', expr, coyote_ast.Var('0')):
                    if debug: print(f'{circuit} removing +0')
                    return optimize_circuit(expr, debug=debug)
                case ('*', coyote_ast.Var('0'), expr) | ('*', expr, coyote_ast.Var('0')):
                    if debug: print(f'{circuit} removing *0')
                    return coyote_ast.Var('0')
                case ('*', coyote_ast.Var('1'), expr) | ('*', expr, coyote_ast.Var('1')):
                    if debug: print(f'{circuit} removing *1')
                    return optimize_circuit(expr, debug=debug)
                case _:
                    # circuit.lhs = optimize_circuit(lhs, debug=debug)
                    # circuit.rhs = optimize_circuit(rhs, debug=debug)
                    # return circuit
                    return coyote_ast.Op(op, optimize_circuit(lhs), optimize_circuit(rhs))

    raise TypeError(type(circuit))


def codegen_scalar(circuit: num_array):
    compiler = CompilerV2()
    make_op = lambda e: e if isinstance(e, coyote_ast.Op) else coyote_ast.Op('~', e, e)
    outputs: list[list[int]] = [[cast(int, compiler.compile(make_op(bit)).val) for bit in n.bits] for n in circuit.nums]
    print(f'{len(compiler.code)} instructions')
    print('\n'.join(map(str, compiler.code)))
    generated_lines = []
    
    aliases: dict[int, str] = {} # register number to the C++ name
    loaded_numbers: dict[str, int] = {} # input name to the wire index
    
    # maps each plaintext/ciphertext physical register to the temporary register its currently holding
    ptxt_allocated: dict[int, int] = {}
    ctxt_allocated: dict[int, int] = {}
    end_liveness: dict[int, int] = defaultdict(int) # which instruction each temporary register stops being live
    
    for instr in reversed(compiler.code):
        if instr.lhs.reg and instr.lhs.val not in end_liveness:
            end_liveness[cast(int, instr.lhs.val)] = cast(int, instr.dest.val)
        if instr.rhs.reg and instr.rhs.val not in end_liveness:
            end_liveness[cast(int, instr.rhs.val)] = cast(int, instr.dest.val)
    
    for out in sum(outputs, []):
        end_liveness[out] = len(compiler.code)
    
    parts: list[list[str]] = []
    
    def reg2var(reg: int) -> str:
        return aliases[reg]
    
    def atom2name(atom: coyote_ast.Atom) -> str:
        if atom.reg:
            return reg2var(cast(int, atom.val))
        assert isinstance(atom.val, str)
        if atom.val.isnumeric():
            return f'NTL::ZZX({atom.val})'
        if atom.val not in loaded_numbers:
            loaded_numbers[atom.val] = max(loaded_numbers.values(), default=-1) + 1
        return f'inputs[{loaded_numbers[atom.val]}]'
    
    def assign(dest: coyote_ast.Atom, value: coyote_ast.Atom):
        vec, allocator = ('ptxt_regs', ptxt_allocated) if is_ptxt(value) else ('ctxt_regs', ctxt_allocated)
        for register, temp in allocator.items():
            if cast(int, dest.val) > end_liveness[temp]:
                # `register` is holding a dead value, kick it out
                allocator[register] = cast(int, dest.val)
                generated_lines.append(f'{vec}[{register}] = {atom2name(value)};')
                aliases[cast(int, dest.val)] = f'{vec}[{register}]'
                break
        else:
            # couldn't find any temporaries to kick out, add a new register
            new_register = max(allocator.keys(), default=-1) + 1
            allocator[new_register] = cast(int, dest.val)
            generated_lines.append(f'{vec}.push_back({atom2name(value)});')
            aliases[cast(int, dest.val)] = f'{vec}[{new_register}]'
                
        
        # # find a register to put it on:
        # # 1. check if any allocated registers are dead
        # for register, temp in ptxt_allocated.items():
        #     if cast(int, dest.val) < end_liveness[temp]:
        #         # kick out `temp`, allocate `value` on `register`
        #         ptxt_allocated[register] = cast(int, dest.val)
        # else:
        #     pass
        
        # vec, count = ('ptxt_regs', num_ptxts) if is_ptxt(value) else ('ctxt_regs', num_ctxts)
        # generated_lines.append(f'{vec}.push_back({atom2name(value)});')
        # aliases[cast(int, dest.val)] = f'{vec}[{count}]'
        
        # num_ptxts += is_ptxt(value)
        # num_ctxts += not is_ptxt(value)

    plaintext_regs: set[int] = set() # which registers hold a plaintext value

    is_ptxt = lambda atom: atom.val in plaintext_regs or atom2name(atom).startswith('NTL')

    for i, instr in enumerate(compiler.code):
        if i and i % 1000 == 0:
            parts.append(generated_lines)
            generated_lines = []
        assert isinstance(instr.dest.val, int)
        if instr.op == '~':
            assert instr.lhs == instr.rhs # sanity check
            aliases[instr.dest.val] = atom2name(instr.lhs)
            if aliases[instr.dest.val].startswith('NTL'):
                plaintext_regs.add(instr.dest.val)
        elif instr.op in '+*':
            # if one operand is ptxt and the other is ctxt, load the ctxt first
            if is_ptxt(instr.lhs) and not is_ptxt(instr.rhs):
                instr.lhs, instr.rhs = instr.rhs, instr.lhs
                
            # generated_lines += [f'auto {atom2name(instr.dest)} = {atom2name(instr.lhs)};']
            assign(instr.dest, instr.lhs)
                
            if is_ptxt(instr.rhs):
                func = 'addConstant' if instr.op == '+' else 'multByConstant'
                generated_lines += [f'{atom2name(instr.dest)}.{func}({atom2name(instr.rhs)});']
            else:
                generated_lines += [f'{atom2name(instr.dest)} {instr.op}= {atom2name(instr.rhs)};']
                
            # if both operands are ptxt, the result is ptxt
            if is_ptxt(instr.lhs) and is_ptxt(instr.rhs):
                plaintext_regs.add(instr.dest.val)
            
        else:
            raise TypeError(instr.op)
        
        if not is_ptxt(instr.dest):
            generated_lines += [f'show({reg2var(cast(int, instr.dest.val))}, {instr.dest.val});']
    parts.append(generated_lines)
    part_template = """
void part{n}(EncInfo &info, std::vector<ctxt_bit> inputs, std::vector<ctxt_bit> &ctxt_regs, std::vector<zzx_vec> &ptxt_regs)
{{
    {code}
}}
"""


    part_code = '\n\n'.join([part_template.format(n=i, code='\n    '.join(part)) for i, part in enumerate(parts)])
            
    return_values = [f'{{{", ".join([reg2var(bit) for bit in output])}}}' for output in outputs]
    generated_lines = [
        'std::vector<ctxt_bit> ctxt_regs;',
        'std::vector<zzx_vec> ptxt_regs;'
    ]
    for i in range(len(parts)):
        generated_lines.append(f'part{i}(info, inputs, ctxt_regs, ptxt_regs);')

    
    generated_lines += [
        f'return std::vector<std::vector<ctxt_bit>>{{ {", ".join([val for val in return_values])} }};'
    ]
    
    
    
    
    generated_code = '\n    '.join(generated_lines)
    
    prepare_lines = ['std::vector<ctxt_bit> wires;']
    for name in sorted(loaded_numbers.keys(), key=lambda name: loaded_numbers[name]):
        prepare_lines += [f'wires.push_back(encrypt_vector(info, ptxt{{inputs["{name}"]}}));']
    
    prepare_lines += ['return wires;']    
    
    prepare_code = '\n    '.join(prepare_lines)
        
    return f"""
#include <coyote-runtime.hpp>

#ifdef DEBUG
#define show(x, n) _show(info, x, #n)
#else
#define show(x, n)
#endif

void _show(EncInfo &info, ctxt_bit vec, std::string name)
{{
    std::cout << "r" << name << " = " << decrypt_vector(info, vec)[0] << "\\n";
}}

{part_code}

std::vector<ctxt_bit> Prepare(EncInfo &info, std::unordered_map<std::string, int> inputs)
{{
    {prepare_code}
}}

std::vector<std::vector<ctxt_bit>> Compute(EncInfo &info, std::vector<ctxt_bit> inputs)
{{
    {generated_code}
}} 
"""
            


def codegen_mux(circuits: num_array, scalar=False):
    compiler = CompilerV2()
    outputs: list[list[int]] = [[cast(int, compiler.compile(bit).val) for bit in circuit.bits] for circuit in circuits.nums]
    print(f'Outputs: {outputs}')
    # input()
    print(f'{len(compiler.code)} instructions')

    # path = input('cached path? ')
    # path = 'mux_schedules/mux_schedules/associative_array'
    # data = open(path).read().replace('lanes:', '').replace('alignment:', '\n').split('\n')
    # # print(data)
    # lanes = eval(data[0])
    # alignment = eval(data[-1])
    # assert len(lanes) == len(alignment) and len(alignment) == len(compiler.code), (len(lanes), len(alignment), len(compiler.code))
    # schedule = Schedule(lanes, alignment, compiler.code)
    # result = vectorize_cached(schedule)
    result = vectorize(compiler, output_groups=[{*outs} for outs in outputs], search_rounds=200)
    
    
    vectorized_code = result.instructions
    lanes = result.lanes
    align = result.alignment
    
    rets: list[int] = []
    widths = []
    max_reg = max(align)
    
    for outs in outputs:
        vec_outputs, liveness, width = analyzeV2(lanes, align, {*outs})
        widths.append(width)
        def mask(s): return [int(i in s) for i in range(width)]
        ret = next(iter(vec_outputs))

        if len(vec_outputs) > 1:
            vectorized_code.append(VecBlendInstr(f'__v{max_reg + 1}', [f'__v{out}' for out in vec_outputs], [mask(liveness[out]) for out in vec_outputs]))
            # ret = max(vec_outputs) + 1
        else:
            vectorized_code.append(VecLoadInstr(f'__v{max_reg + 1}', f'__v{next(iter(vec_outputs))}'))
        rets.append(max_reg + 1)
        liveness[max_reg + 1] = set().union(*(liveness[o] for o in vec_outputs))
        max_reg += 1
            
        

    
    # vouts: set[int] = set()
    # for out in outputs:
    #     vouts.add(align[out])
    # vout, = vouts # there should only be one output vector
    print(f'Outputs: {outs}')
    print(f'Output lanes: {[lanes[out] for out in outs]}')
    
    return vectorized_code, rets, [[lanes[out] for out in outs] for outs in outputs], result
