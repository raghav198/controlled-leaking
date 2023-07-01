from dataclasses import dataclass
from typing import Callable

from coyote import coyote_ast
from coyote.codegen import (VecBlendInstr, VecConstInstr, VecInstr,
                            VecLoadInstr, VecOpInstr, VecRotInstr)
from coyote.coyote_ast import CompilerV2
from coyote.vectorize_circuit import vectorize

from holla import (ChallahArithExpr, ChallahArray, ChallahBranch, ChallahLeaf,
                   ChallahTree, ChallahVar)

bit = coyote_ast.Expression

bitwidth = 2


@dataclass
class num:
    """Bits of a number represented least significant bit first (e.g. `12` as a four bit number is `0011`)"""
    bits: list[bit]

    def __post_init__(self):
        assert len(self.bits) <= bitwidth
        while len(self.bits) < bitwidth:
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
        val += 2 ** (bitwidth - 1)
    assert val < 2 ** bitwidth, f'Value {val} too big for {bitwidth} bits'
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
        carry = left_bit * right_bit
    return num(sum_bits)


def mul(left: num, right: num) -> num:
    product_bits: list[bit] = []

    for k in range(bitwidth):
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
            return num([coyote_ast.Var(f'{name}.{b}') for b in range(bitwidth)])
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


def to_cpp(code: list[VecInstr]):

    def reg2var(reg: str) -> str:
        if reg in aliases:
            return aliases[reg]
        return reg[2:]

    aliases: dict[str, str] = {}
    cpp: list[str] = []
    prep_code: list[str] = []
    all_masks: set[str] = set()
    for line in code:
        match line:
            case VecRotInstr(dest, operand, shift):
                cpp += [
                    f'ctxt_bit {reg2var(dest)} = {reg2var(operand)};',
                    f'info.ea.rotate({reg2var(dest)}, {shift});'
                ]
            case VecOpInstr(dest, op, lhs, rhs):
                
                if op == '-':
                    op = '+'
                    print('[WARN] No subtraction in mod 2 >:(')
                
                if reg2var(lhs).startswith('plaintexts') and reg2var(rhs).startswith('plaintexts'):
                    cpp += [
                        f'plaintexts["{dest}"] = {reg2var(lhs)};',
                        f'plaintexts["{dest}"] {op}= {reg2var(rhs)};'
                    ]
                    aliases[dest] = f'plaintexts["{dest}"]'
                    continue
                
                if reg2var(lhs).startswith('plaintexts') and not reg2var(rhs).startswith('plaintexts'):
                    lhs, rhs = rhs, lhs
                
                cpp += [
                    f'ctxt_bit {reg2var(dest)} = {reg2var(lhs)};',
                    f'{reg2var(dest)}.multiplyBy({reg2var(rhs)});' if op == '*' else f'{reg2var(dest)} {op}= {reg2var(rhs)};'
                ]
            case VecLoadInstr(dest, src):
                aliases[dest] = src
            case VecConstInstr(dest, vals):
                ptxt_vec = [f'inputs["{val}"]' if not val.isnumeric() else val for val in vals]
                to_encrypt = any(not val.isnumeric() for val in vals)
                prep_dest = 'ciphertexts' if to_encrypt else 'plaintexts'
                encoder = 'encrypt_vector' if to_encrypt else 'encode_vector'
                prep_code += [
                    f'ptxt_bit {reg2var(dest)}{{{", ".join(ptxt_vec)}}}',
                    f'{prep_dest}["{reg2var(dest)}"] = {encoder}(info, {reg2var(dest)});'
                ]
                
                aliases[dest] = f'{prep_dest}["{reg2var(dest)}"]'
            case VecBlendInstr(dest, vals, masks):
                all_masks |= {'"' + ''.join(map(str, mask)) + '"' for mask in masks}
                args = [f'{{{reg2var(val)}, masks["{"".join(map(str, mask))}"]}}' for val, mask in zip(vals, masks)]
                cpp += [
                    f'ctxt_bit {reg2var(dest)} = blend({{{", ".join(args)}}});'
                ]
            case _:
                raise TypeError(type(line))
            

    prep_code += [f'masks["{mask}"] = make_mask(info, "{mask}");' for mask in all_masks]
    
    return prep_code, cpp

            


def codegen_mux(circuit: num):
    compiler = CompilerV2()
    for bit in circuit.bits:
        compiler.compile(bit)
    vectorized_code = vectorize(compiler)
    return vectorized_code
