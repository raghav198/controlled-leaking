from __future__ import annotations
from dataclasses import dataclass
from functools import reduce
from typing import Callable

from pita import PitaArithExpr, PitaArrayExpr, PitaCondExpr, PitaExpr, PitaFuncCallExpr, PitaFuncDefExpr, PitaIndexExpr, PitaLetExpr, PitaNumExpr, PitaVarExpr, gcd, pita_map, pita_num_map

@dataclass
class ChallahVar:
    name: str

    def __repr__(self):
        return self.name
    
    def normalize(self):
        return self

@dataclass
class ChallahArithExpr:
    # check sizes
    def __post_init__(self):
        assert isinstance(self.left, ChallahArray) == isinstance(self.right, ChallahArray)
        if isinstance(self.left, ChallahArray) and isinstance(self.right, ChallahArray):
            assert len(self.left.elems) == len(self.right.elems)
    
    left: ChallahLeaf
    op: str
    right: ChallahLeaf
    
    def __repr__(self):
        return f'({self.left} {self.op} {self.right})'
    
    def normalize(self) -> ChallahArray | ChallahArithExpr:
        if isinstance(self.left, ChallahArray) and isinstance(self.right, ChallahArray):
            return ChallahArray(elems=[ChallahArithExpr(left.normalize(), self.op, right.normalize()) for left, right in zip(self.left.elems, self.right.elems)])
        return self
    
    
@dataclass
class ChallahArray:
    elems: list[ChallahVar | ChallahArithExpr]
    
    def __repr__(self):
        return repr(self.elems)
    
    def normalize(self):
        return self

@dataclass
class ChallahBranch:
    left: ChallahLeaf
    right: ChallahLeaf
    true: ChallahTree
    false: ChallahTree
    
    def __repr__(self):
        return f'if ({self.left} < {self.right}) {{\n{self.true}\n}} else {{\n{self.false}\n}}'
    
    def normalize(self):
        true = [t.normalize() for t in self.true] if isinstance(self.true, list) else self.true.normalize()
        false = [t.normalize() for t in self.false] if isinstance(self.false, list) else self.false.normalize()
        return ChallahBranch(self.left, self.right, true, false)

ChallahLeaf = ChallahVar | ChallahArithExpr | ChallahArray
ChallahTree = ChallahLeaf | list[ChallahLeaf] | ChallahBranch

def pprint(tree: ChallahTree, depth: int = 0, start=''):
    if isinstance(tree, ChallahBranch):
        print(f'{" " * depth * 4}{start}{tree.left} < {tree.right}')
        pprint(tree.true, depth + 1, start='\u2713 ')
        pprint(tree.false, depth + 1, start='\u2717 ')
    else:
        print(' ' * depth * 4 + start + str(tree))

def operad_compose(left: ChallahTree, right: ChallahTree, f: Callable[[ChallahLeaf, ChallahLeaf], ChallahTree]):
    if isinstance(left, ChallahLeaf) and isinstance(right, ChallahLeaf):
        return f(left, right)
    if isinstance(left, ChallahBranch):
        return ChallahBranch(left.left, left.right, operad_compose(left.true, right, f), operad_compose(left.false, right, f))
    if isinstance(right, ChallahBranch):
        return ChallahBranch(right.left, right.right, operad_compose(left, right.true, f), operad_compose(left, right.false, f))
    raise TypeError(type(left), type(right))


def combine_trees(left: ChallahTree, op: str, right: ChallahTree):
    return operad_compose(left, right, lambda l, r: ChallahArithExpr(l, op, r))


def compare_trees(left: ChallahTree, right: ChallahTree, true: ChallahTree, false: ChallahTree):
    return operad_compose(left, right, lambda l, r: ChallahBranch(l, r, true, false))


def concat(left: ChallahLeaf, right: ChallahLeaf) -> ChallahArray:
    def arr(leaf: ChallahLeaf) -> ChallahArray:
        match leaf:
            case ChallahArray(elems):
                return leaf
            case ChallahVar(_) | ChallahArithExpr(_, _, _):
                return ChallahArray(elems=[leaf]) 
    
    return ChallahArray(elems=arr(left).elems + arr(right).elems)
        
        
def concatenate_trees(left: ChallahTree, right: ChallahTree):
    return operad_compose(left, right, concat)


def desugar_indices(expr: PitaExpr):
    match expr:
        case PitaIndexExpr(arr, index):
            return desugar_indices(arr.elems[index])
        case _:
            return pita_map(expr, desugar_indices)


# does not work for inlining function calls!
# call this from the inside out!
# def substitute(var: str, val: PitaNumExpr, body: PitaNumExpr) -> PitaNumExpr:
#     match body:
#         case PitaVarExpr(name):
#             if name == var:
#                 return val
#             return body
#         case PitaArithExpr(left, op, right):
#             return PitaArithExpr(substitute(var, val, left), op, substitute(var, val, right))
#         case PitaLetExpr(var2, val2, body2):
#             return PitaLetExpr(var2, val2, substitute(var, val, body2))
#         case PitaCondExpr(left, right, true, false):
#             return PitaCondExpr(substitute(var, val, left), substitute(var, val, right), substitute(var, val, true), substitute(var, val, false))
#         case PitaFuncCallExpr(name, params):
#             return PitaFuncCallExpr(name, [substitute(var, val, param) for param in params])
#         case PitaArrayExpr(elems):
#             return PitaArrayExpr([substitute(var, val, elem) for elem in elems])
#     raise TypeError(type(body))

def substitute(var: str, val: PitaNumExpr, body: PitaNumExpr) -> PitaNumExpr:
    match body:
        case PitaVarExpr(name):
            if name == var:
                return val
            return body
        case _:
            return pita_num_map(body, lambda e: substitute(var, val, e))

# call this from the inside out!
# def inline(func: str, args: list[str], body: PitaNumExpr, prog: PitaNumExpr):
#     match prog:
#         case PitaVarExpr(name):
#             return PitaVarExpr(name)
#         case PitaArithExpr(left, op, right):
#             return PitaArithExpr(inline(func, args, body, left), op, inline(func, args, body, right))
#         case PitaLetExpr(var, val, body2):
#             assert isinstance(val, PitaNumExpr), f'Cannot inline inside function definition {var}!'
#             return PitaLetExpr(var, inline(func, args, body, val), inline(func, args, body, body2))
            
#         case PitaCondExpr(left, right, true, false):
#             return PitaCondExpr(inline(func, args, body, left), inline(func, args, body, right), inline(func, args, body, true), inline(func, args, body, false))
#         case PitaFuncCallExpr(name, params):
#             if name == func:
#                 evaluated_body = body
#                 for arg, param in zip(args, params):
#                     evaluated_body = substitute(arg, param, evaluated_body)
#                 return evaluated_body
#             return PitaFuncCallExpr(name, params)
#     raise TypeError(type(prog))
def inline(func: str, args: list[str], body: PitaNumExpr, prog: PitaNumExpr):
    match prog:
        case PitaFuncCallExpr(name, params):
            if name == func:
                evaluated_body = body
                for arg, param in zip(args, params):
                    evaluated_body = substitute(arg, param, evaluated_body)
                return inline(func, args, body, evaluated_body)
            return PitaFuncCallExpr(name, params)
        case _:
            return pita_map(prog, lambda e: inline(func, args, body, e)) # type: ignore


# def inline_all(expr: PitaExpr):
#     match expr:
#         case PitaVarExpr(name):
#             return PitaVarExpr(name)
#         case PitaArithExpr(left, op, right):
#             return PitaArithExpr(inline_all(left), op, inline_all(right))
#         case PitaLetExpr(var, val, body):
#             match val:
#                 case PitaFuncDefExpr(args, func_body):
#                     return inline(var, args, func_body, inline_all(body))
#                 case _:
#                     return PitaLetExpr(var, inline_all(val), inline_all(body))
#         case PitaCondExpr(left, right, true, false):
#             return PitaCondExpr(inline_all(left), inline_all(right), inline_all(true), inline_all(false))
#         case PitaFuncCallExpr(name, params):
#             return PitaFuncCallExpr(name, [inline_all(param) for param in params])
#         case PitaArrayExpr(elems):
#             return PitaArrayExpr([inline_all(elem) for elem in elems])
#     raise TypeError(type(expr))

def inline_all(expr: PitaExpr):
    match expr:
        case PitaLetExpr(var, val, body):
            match val:
                case PitaFuncDefExpr(args, func_body):
                    inlined_body = inline_all(body)
                    print('original body:')
                    print(body)
                    print('inlined body:')
                    print(inlined_body)
                    post_inlining = inline(var, args, func_body, inlined_body)
                    print('post inlining:')
                    print(post_inlining)
                    return post_inlining
                case _:
                    return PitaLetExpr(var, inline_all(val), inline_all(body))
        case _:
            return pita_map(expr, inline_all)

# expects all function definitions and calls to be inlined
# def substitute_all(expr: PitaExpr):
#     match expr:
#         case PitaVarExpr(name):
#             return PitaVarExpr(name)
#         case PitaArithExpr(left, op, right):
#             return PitaArithExpr(substitute_all(left), op, substitute_all(right))
#         case PitaLetExpr(var, val, body):
#             assert isinstance(val, PitaNumExpr), f'Function definition {var} not inlined before substitution!'
#             return substitute(var, val, substitute_all(body))
#         case PitaCondExpr(left, right, true, false):
#             return PitaCondExpr(substitute_all(left), substitute_all(right), substitute_all(true), substitute_all(false))
#     raise TypeError(type(expr))

def substitute_all(expr: PitaExpr):
    match expr:
        case PitaLetExpr(var, val, body):
            assert isinstance(val, PitaNumExpr), f'Function definition {var} not inlined before substitution!'
            return substitute(var, val, substitute_all(body))
        case _:
            return pita_num_map(expr, substitute_all)

def comparison_folding(tree: ChallahTree, known: list[tuple[ChallahLeaf, ChallahLeaf]] = []):
    match tree:
        case ChallahBranch(left, right, true, false):
            if (left, right) in known:
                return comparison_folding(true, known)
            if (right, left) in known:
                return comparison_folding(false, known)
            return ChallahBranch(left, right, comparison_folding(true, known + [(left, right)]), comparison_folding(false, known + [(right, left)]))
        case _:
            return tree


def treeify_expr(expr: PitaExpr):
    match expr:
        case PitaVarExpr(name):
            return ChallahVar(name)
        case PitaArithExpr(left, op, right):
            ltree = treeify_expr(left)
            rtree = treeify_expr(right)
            
            return comparison_folding(combine_trees(ltree, op, rtree))
            
        case PitaCondExpr(left, right, true, false):
            return comparison_folding(compare_trees(treeify_expr(left), treeify_expr(right), treeify_expr(true), treeify_expr(false)))
        case PitaArrayExpr(elems):
            tree_elems = [treeify_expr(elem) for elem in elems]
            return comparison_folding(reduce(concatenate_trees, tree_elems[1:], tree_elems[0]))
    raise TypeError(type(expr))


passes = [inline_all, substitute_all, desugar_indices, treeify_expr, comparison_folding, lambda t: t.normalize()]

def compile(prog) -> ChallahTree:
    print(prog)
    input()
    for p in passes:
        prog = p(prog)
        print(prog)
        input()
    return prog

if __name__ == '__main__':
    prog = gcd(PitaVarExpr('a'), PitaVarExpr('b'), 8)
    print(prog)
    print('-' * 10)
    from time import time
    for p in passes:
        print(f'Running pass `{str(p.__name__)}`...')
        s = time()
        prog = p(prog)
        print(f'({int(1000*(time() - s))} ms)')
    print(prog)
