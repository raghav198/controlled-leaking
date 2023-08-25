from collections import defaultdict
from math import ceil
from typing import cast

from coyote import coyote_ast
from coyote.codegen import (Schedule, VecBlendInstr, VecConstInstr,
                            VecLoadInstr, VecOpInstr, VecRotInstr, codegen)
from coyote.coyote_ast import CompilerV2
from coyote.vectorize_circuit import CodeObject, vectorize

from holla import (ChallahArithExpr, ChallahArray, ChallahBranch, ChallahLeaf,
                   ChallahTree, ChallahVar)


def parse_circuit(expr: ChallahLeaf) -> coyote_ast.Expression:
    match expr:
        case ChallahVar(name):
            return coyote_ast.Var(name)
        case ChallahArithExpr(left, op, right):
            return coyote_ast.Op(op, parse_circuit(left), parse_circuit(right))
    raise TypeError(type(expr))


def parse_decision_circuits(tree: ChallahTree) -> list[tuple[coyote_ast.Expression, coyote_ast.Expression, bool]]:
    if isinstance(tree, ChallahBranch):
        return parse_decision_circuits(tree.true) + parse_decision_circuits(tree.false) + [(parse_circuit(tree.left), parse_circuit(tree.right), tree.lt)]
    return []


def parse_label_circuits(tree: ChallahTree) -> list[list[coyote_ast.Expression]]:
    if isinstance(tree, ChallahBranch):
        return parse_label_circuits(tree.true) + parse_label_circuits(tree.false)
    if isinstance(tree, ChallahArray):
        return [[parse_circuit(elem) for elem in tree.elems]]
    if isinstance(tree, ChallahLeaf):
        return [[parse_circuit(tree)]]
    raise TypeError(type(tree))


def analyzeV2(lanes: list[int], schedule: list[int], outputs: set[int]):
    vec_outputs: set[int] = {schedule[o] for o in outputs}
    def out_live(n): return {lanes[o] for o in outputs if schedule[o] == n}
    liveness = {o: out_live(o) for o in vec_outputs}

    return list(vec_outputs), liveness, max(lanes) + 1


def compile_circuits_stupid(circuits: list[coyote_ast.Expression]):
    if not circuits:
        return CodeObject(), None, 0
    
    compiler = CompilerV2()
    
    outputs: list[int] = []
    for expr in circuits:
        if isinstance(expr, coyote_ast.Op):
            outputs.append(cast(int, compiler.compile(expr).val))
        else:
            outputs.append(cast(int, compiler.compile(coyote_ast.Op('~', expr, expr)).val))

    alignment: list[int] = list(range(len(compiler.code)))
    lanes: list[int] = []
    cur_lane = 0
    for i in range(len(compiler.code)):
        lanes.append(cur_lane)
        if i in outputs: cur_lane += 1
        
    print('scalar:')
    print('\n'.join(map(str, compiler.code)))
    print(f'outputs: {outputs}')
    print(f'alignment: {alignment}')
    print(f'lanes: {lanes}')
    # input()
        
    code = codegen(Schedule(lanes, alignment, compiler.code))
    code.append(VecBlendInstr(f'__v{max(outputs) + 1}', [f'__v{out}' for out in outputs], [[int(i == j) for i in range(len(outputs))] for j in range(len(outputs))]))
    print('\n'.join(map(str, code)))
    # input()
    return CodeObject(instructions=code, lanes=lanes, alignment=alignment, vector_width=len(outputs)), max(outputs) + 1, len(circuits)

def compile_circuits(circuits: list[coyote_ast.Expression], rounds: int):
    if not circuits:
        return CodeObject(), None, 0
    # all the plain variables should be grouped
    group = {v.name for v in circuits if isinstance(v, coyote_ast.Var)}
    comp = CompilerV2(input_groups=[group]) if group else CompilerV2(input_groups=[])

    force_lanes = {}
    for lane, expr in enumerate(circuits):
        if isinstance(expr, coyote_ast.Op):
            force_lanes[comp.compile(expr).val] = lane
        else:
            force_lanes[comp.compile(coyote_ast.Op('~', expr, expr)).val] = lane

    assert len(force_lanes.keys()) == len(circuits)

    # input("\n".join(map(str, comp.code)))

    result = vectorize(comp, extra_force_lanes=force_lanes, search_rounds=rounds)
    # input()
    
    # code = result.code
    # lanes = result.lanes
    # alignment = result.alignment

    outputs, liveness, width = analyzeV2(result.lanes, result.alignment, set(force_lanes.keys()))
    def mask(s): return [int(i in s) for i in range(width)]
    ret = next(iter(outputs))

    if len(outputs) > 1:
        result.instructions.append(VecBlendInstr(f'__v{max(outputs) + 1}', [f'__v{out}' for out in outputs], [mask(liveness[out]) for out in outputs]))
        ret = max(outputs) + 1
    liveness[max(outputs) + 1] = set().union(*(liveness[o] for o in outputs))

    return result, ret, len(circuits)


def compile_array_circuits_stupid(circuits: list[list[coyote_ast.Expression]]):
    compiler = CompilerV2()
    array_idx_outputs: dict[int, list[int]] = defaultdict(list)
    all_outputs: list[int]
    
    for lane, arr in enumerate(circuits):
        out_group: set[int] = set()
        for idx, expr in enumerate(arr):
            reg: int
            if isinstance(expr, coyote_ast.Op):
                reg = cast(int, compiler.compile(expr).val)
            else:
                reg = cast(int, compiler.compile(coyote_ast.Op('~', expr, expr)))
                
            array_idx_outputs[idx].append(reg)
            out_group.add(reg)
            

    alignment: list[int] = list(range(len(compiler.code)))
    lanes: list[int] = []
    for idx in array_idx_outputs:
        cur_lane = 0
        for i in range(len(lanes), array_idx_outputs[idx][-1] + 1):
            lanes.append(cur_lane)
            if i in array_idx_outputs[idx]: cur_lane += 1
            
    code = codegen(Schedule(lanes, alignment, compiler.code))
    next_ret = len(alignment)
    rets: list[int] = []
    for idx in array_idx_outputs:
        outputs = array_idx_outputs[idx]
        code.append(VecBlendInstr(f'__v{next_ret}', [f'__v{out}' for out in outputs], [[int(i == j) for i in range(len(outputs))] for j in range(len(outputs))]))
        rets.append(next_ret)
        next_ret += 1
        
    print('\n'.join(map(str, code)))
    # input()
        
    return CodeObject(instructions=code, lanes=lanes, alignment=alignment, vector_width=max(map(len, array_idx_outputs.values()))), rets, len(circuits)
    
    
            

def compile_array_circuits(circuits: list[list[coyote_ast.Expression]], rounds: int):
    group = {v.name for c in circuits for v in c if isinstance(v, coyote_ast.Var)}
    # group = set()
    comp = CompilerV2(input_groups=[group]) if group else CompilerV2(input_groups=[])
    
    force_lanes = {}
    array_idx_outputs: dict[int, list[int]] = defaultdict(list)

    for lane, arr in enumerate(circuits):
        out_group: set[int] = set()
        for idx, expr in enumerate(arr):
            if isinstance(expr, coyote_ast.Op):
                force_lanes[reg := comp.compile(expr).val] = lane
            else:
                force_lanes[reg := comp.compile(coyote_ast.Op('~', expr, expr)).val] = lane
            assert isinstance(reg, int)
            array_idx_outputs[idx].append(reg)
            out_group.add(reg)

    assert len(force_lanes.keys()) == sum(map(len, circuits))

    # input('\n'.join(map(str, comp.code)))
    comp.input_groups = []
    code = vectorize(comp, extra_force_lanes=force_lanes,
                     output_groups=list(map(set, array_idx_outputs.values())), search_rounds=rounds)
    
    # input('\n'.join(map(str, code.instructions)))
    
    # code = result.code
    # lanes = result.lanes
    # alignment = result.alignment
    
    rets: list[int] = []
    widths = []

    max_reg = max(code.alignment)

    for idx in array_idx_outputs:
        outputs, liveness, width = analyzeV2(code.lanes, code.alignment, set(array_idx_outputs[idx]))
        print(f'array[{idx}] on schedule slot(s) {outputs}')
        widths.append(width)
        def mask(s): return [int(i in s) for i in range(width)]
        # ret = next(iter(outputs))
        if len(outputs) > 1:
            code.instructions.append(VecBlendInstr(f'__v{max_reg + 1}', [f'__v{out}' for out in outputs], [mask(liveness[out]) for out in outputs]))
        else:
            code.instructions.append(VecLoadInstr(f'__v{max_reg + 1}', f'__v{next(iter(outputs))}'))
        rets.append(max_reg + 1)
        max_reg += 1
        
    return code, rets, len(circuits)


def vectorize_decisions(tree: ChallahTree, rounds: int):
    try:
        lefts, rights, lt_mask = cast(tuple[list[coyote_ast.Expression], list[coyote_ast.Expression], list[bool]], zip(*parse_decision_circuits(tree)))
    except ValueError:
        lefts = []
        rights = []
        lt_mask = []

    eq_mask = list(map(lambda b: not b, lt_mask))

    left_code, left_vec, left_width = compile_circuits(lefts, rounds)
    right_code, right_vec, right_width = compile_circuits(rights, rounds)

    left_cpp = generate_coyote_kernel(left_code, f'COILLeftKernel', left_width, [f'__v{left_vec}'] if left_vec else [])
    right_cpp = generate_coyote_kernel(right_code, f'COILRightKernel', right_width, [f'__v{right_vec}'] if right_vec else [])

    return left_cpp, right_cpp, ''.join(map(str, map(int, lt_mask))), ''.join(map(str, map(int, eq_mask)))


def vectorize_labels(tree: ChallahTree, rounds: int):
    label_circuits = parse_label_circuits(tree)
    # label_circuits = []

    # for slice in zip(*parse_label_circuits(tree)):
    #     label_circuits += list(slice)

    code, vecs, width = compile_array_circuits(label_circuits, rounds)
    # print('\n'.join(map(str, code)))
    return generate_coyote_kernel(code, f'COILLabels', width, [f'__v{vec}' for vec in vecs])



def generate_coyote_cpp(code: CodeObject, aliases: dict[str, str]={}):
    op2func: dict[str, str] = {'+': 'add', '*': 'mul', '-': 'sub'}
    
    def reg2var(reg: str) -> str:
        if reg in aliases:
            return aliases[reg]
        return reg[2:]
    
    generated_code: list[str] = []
    for line in code.instructions:
        match line:
            case VecRotInstr(dest, operand, shift):
                generated_code.append(f'auto {reg2var(dest)} = rotate({reg2var(operand)}, {shift});')
            case VecOpInstr(dest, op, lhs, rhs):
                generated_code.append(f'auto {reg2var(dest)} = {op2func[op]}({reg2var(lhs)}, {reg2var(rhs)});')
            case VecLoadInstr(dest, src):
                aliases[dest] = reg2var(src)
            case VecConstInstr(_, _):
                pass
            case VecBlendInstr(dest, vals, masks):
                args = [f'{{{reg2var(val)}, masks["{"".join(map(str, mask))}"]}}' for val, mask in zip(vals, masks)]
                generated_code.append(f'auto {reg2var(dest)} = blend({{{", ".join(args)}}});')
    return generated_code


def generate_coyote_kernel(code: CodeObject, name: str, width: int, output: list[str]):

    def reg2var(reg: str) -> str:
        if reg in input_wire_mapping:
            return f'input_wires[{input_wire_mapping[reg]}]'
        if reg in aliases:
            return aliases[reg]
        return reg[2:]

    positive_pads = max(0, ceil(code.max_rotation / code.vector_width))
    negative_pads = max(0, -ceil(code.min_rotation / code.vector_width))
    
    if not name.endswith('Labels'): # Left and Right kernels need an extra negative pad to prepare for COPSE
        negative_pads += 1

    input_wire_mapping: dict[str, int] = {}  # register -> input wire number
    aliases: dict[str, str] = {}  # variable before inlining -> variable after inlining

    input_wire_loading: list[str] = []
    masks = set()
    for line in code.instructions:
        if isinstance(line, VecConstInstr):
            ptxt_vec = [f'inputs["{val}"]' if not val.isnumeric() else val for val in line.vals]
            input_wire_loading.append(f'ptxt {reg2var(line.dest)}{{{", ".join(ptxt_vec)}}};')
            input_wire_loading.append(f'input_wires.push_back(encrypt(info, {reg2var(line.dest)}, positive_pads, negative_pads));')
            input_wire_mapping[line.dest] = len(input_wire_mapping)
        elif isinstance(line, VecBlendInstr):
            masks |= {'"' + ''.join(map(str, mask)) + '"' for mask in line.masks}

    input_wire_loading.append(f'add_masks({{{", ".join(masks)}}});')

    aliases.update({reg: f'input_wires[{input_wire_mapping[reg]}]' for reg in input_wire_mapping})

    generated_code = generate_coyote_cpp(code, aliases)

    for out in output:
        generated_code.append(f'output_wires.push_back({reg2var(out)});')

    input_wire_loading_code = '\n        '.join(input_wire_loading)
    kernel_code = '\n        '.join(generated_code)

    cpp = f"""
#include "kernel.hpp"
{name}::{name}(EncInfo& info) : CoyoteKernel(info, {width}) {{}}

void {name}::Prepare(std::unordered_map<std::string, int> inputs) {{
    
    int positive_pads = {positive_pads};
    int negative_pads = {negative_pads};
    
    {input_wire_loading_code}
}}

void {name}::Compute() {{
    {kernel_code}
}}
    """

    return cpp


