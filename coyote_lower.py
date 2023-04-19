from typing import cast
from coyote import coyote_ast, CompilerV2, vectorize
from coyote.codegen import VecBlendInstr, VecLoadInstr, VecInstr, VecRotInstr, VecConstInstr, VecOpInstr
from collections import defaultdict
from holla import ChallahArithExpr, ChallahBranch, ChallahLeaf, ChallahTree, ChallahVar, ChallahArray

def parse_circuit(expr: ChallahLeaf) -> coyote_ast.Expression:
    match expr:
        case ChallahVar(name):
            return coyote_ast.Var(name)
        case ChallahArithExpr(left, op, right):
            return coyote_ast.Op(op, parse_circuit(left), parse_circuit(right))
    raise TypeError(type(expr))
    

def parse_decision_circuits(tree: ChallahTree) -> list[tuple[coyote_ast.Expression, coyote_ast.Expression]]:
    if isinstance(tree, ChallahBranch):
        return parse_decision_circuits(tree.true) + parse_decision_circuits(tree.false) + [(parse_circuit(tree.left), parse_circuit(tree.right))]
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
    out_live = lambda n: {lanes[o] for o in outputs if schedule[o] == n}
    liveness = {o: out_live(o) for o in vec_outputs}
    
    return list(vec_outputs), liveness, max(lanes) + 1
    

def get_outputs(code: list[str]):
    values: dict[str, str] = {}
    outputs: set[str] = set()
    for line in code:
        lhs, rhs = line.split(' = ')
        outputs.add(lhs)
        ops = rhs.replace('+', '|').replace('-', '|').replace('*', '|').replace('>>', '|').replace('blend(', '').replace(', ', ' | ').replace('@', ' | ').split(' | ')
        for op in ops:
            outputs.discard(op)
    return outputs


def compile_circuits(circuits: list[coyote_ast.Expression]):    
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
    
    lanes = []
    sched = []
    code = vectorize(comp, extra_force_lanes=force_lanes, lanes_out=lanes, sched_out=sched)

    outputs, liveness, width = analyzeV2(lanes, sched, set(force_lanes.keys()))
    mask = lambda s: [int(i in s) for i in range(width)]
    ret = next(iter(outputs))
    
    if len(outputs) > 1:
        code.append(VecBlendInstr(f'__v{max(outputs) + 1}', [f'__v{out}' for out in outputs], [mask(liveness[out]) for out in outputs]))
        ret = max(outputs) + 1
    liveness[max(outputs) + 1] = set().union(*(liveness[o] for o in outputs))
    
    return code, ret, len(circuits)


def compile_array_circuits(circuits: list[list[coyote_ast.Expression]]):
    group = {v.name for c in circuits for v in c if isinstance(v, coyote_ast.Var)}
    comp = CompilerV2(input_groups=[group]) if group else CompilerV2(input_groups=[])
    
    force_lanes = {}
    array_idx_outputs = defaultdict(list)
    for lane, arr in enumerate(circuits):
        for idx, expr in enumerate(arr):
            if isinstance(expr, coyote_ast.Op):
                force_lanes[reg := comp.compile(expr).val] = lane
            else:
                force_lanes[reg := comp.compile(coyote_ast.Op('~', expr, expr)).val] = lane
            array_idx_outputs[idx].append(reg)
            
    assert len(force_lanes.keys()) == sum(map(len, circuits))
    
    lanes = []
    sched = []
    code = vectorize(comp, extra_force_lanes=force_lanes, lanes_out=lanes, sched_out=sched)
    print('\n'.join(map(str, code)))
    rets: list[int] = []
    widths = []
    
    max_reg = max(sched)
    print(force_lanes)
    print(lanes)
    print(sched)
    for idx in array_idx_outputs:
        outputs, liveness, width = analyzeV2(lanes, sched, set(array_idx_outputs[idx]))
        print(idx, liveness)
        widths.append(width)
        mask = lambda s: [int(i in s) for i in range(width)]
        # ret = next(iter(outputs))
        if len(outputs) > 1:
            code.append(VecBlendInstr(f'__v{max_reg + 1}', [f'__v{out}' for out in outputs], [mask(liveness[out]) for out in outputs]))
        else:
            code.append(VecLoadInstr(f'__v{max_reg + 1}', f'__v{next(iter(outputs))}'))
        rets.append(max_reg + 1)
        max_reg += 1
    print('\n'.join(map(str, code)))
    return code, rets, len(circuits)

def vectorize_decisions(tree: ChallahTree):
    lefts, rights = cast(list[list[coyote_ast.Expression]], zip(*parse_decision_circuits(tree)))

    left_code, left_vec, left_width = compile_circuits(lefts)
    right_code, right_vec, right_width = compile_circuits(rights)
    
    left_cpp = generate_coyote_cpp(left_code, f'COILLeftKernel', left_width, [f'__v{left_vec}'])
    right_cpp = generate_coyote_cpp(right_code, f'COILRightKernel', right_width, [f'__v{right_vec}'])
    
    return left_cpp, right_cpp


def vectorize_labels(tree: ChallahTree):
    label_circuits = parse_label_circuits(tree)
    # label_circuits = []
    
    # for slice in zip(*parse_label_circuits(tree)):
    #     label_circuits += list(slice)
    
    code, vecs, width = compile_array_circuits(label_circuits)
    print('\n'.join(map(str, code)))
    return generate_coyote_cpp(code, f'COILLabels', width, [f'__v{vec}' for vec in vecs])
    
    
def generate_coyote_cpp(code: list[VecInstr], name: str, width: int, output: list[str]):
    
    def reg2var(reg: str) -> str:
        if reg in input_wire_mapping:
            return f'input_wires[{input_wire_mapping[reg]}]'
        if reg in aliased_variables:
            return aliased_variables[reg]
        return reg[2:]
    
    input_wire_mapping: dict[str, int] = {} # register -> input wire number
    aliased_variables: dict[str, str] = {} # variable before inlining -> variable after inlining
    
    op2func: dict[str, str] = {'+': 'add', '*': 'mul', '-': 'sub'}
    
    input_wire_loading: list[str] = []
    masks = set()
    for line in code:
        if isinstance(line, VecConstInstr):
            ptxt_vec = [f'inputs["{val}"]' if val != '0' else "0" for val in line.vals]
            input_wire_loading.append(f'ptxt {reg2var(line.dest)}{{{", ".join(ptxt_vec)}}};')
            input_wire_loading.append(f'input_wires.push_back(encrypt(info, {reg2var(line.dest)}));')
            input_wire_mapping[line.dest] = len(input_wire_mapping)
        elif isinstance(line, VecBlendInstr):
            masks |= {'"' + ''.join(map(str, mask)) + '"' for mask in line.masks}
    
    input_wire_loading.append(f'add_masks({{{", ".join(masks)}}});')
            
    generated_code: list[str] = []
    for line in code:
        match line:
            case VecRotInstr(dest, operand, shift):
                generated_code.append(f'auto {reg2var(dest)} = rotate({reg2var(operand)}, {shift});')
            case VecOpInstr(dest, op, lhs, rhs):
                generated_code.append(f'auto {reg2var(dest)} = {op2func[op]}({reg2var(lhs)}, {reg2var(rhs)});')
            case VecLoadInstr(dest, src):
                aliased_variables[dest] = reg2var(src)
            case VecConstInstr(_, _):
                pass
            case VecBlendInstr(dest, vals, masks):
                args = [f'{{{reg2var(val)}, masks["{"".join(map(str, mask))}"]}}' for val, mask in zip(vals, masks)]
                generated_code.append(f'auto {reg2var(dest)} = blend({{{", ".join(args)}}});')
                
    for out in output:
        generated_code.append(f'output_wires.push_back({reg2var(out)});')
    
    input_wire_loading_code = '\n        '.join(input_wire_loading)
    kernel_code = '\n        '.join(generated_code)
    
    cpp = f"""
#include "../kernel.hpp"    

struct {name} : public CoyoteKernel {{
    {name}(EncInfo& info) : CoyoteKernel(info, {width}) {{}}
    
    virtual void Prepare(std::unordered_map<std::string, int> inputs) override {{
        {input_wire_loading_code}
    }}
    
    virtual void  Compute() override {{
        {kernel_code}
    }}
}};
    """
    
    return cpp
                
            
            
    
    