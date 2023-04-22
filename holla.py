from __future__ import annotations
from dataclasses import dataclass
from functools import reduce
from typing import Callable
from typecheck import is_ptxt

from pita import PitaArithExpr, PitaArrayExpr, PitaCondExpr, PitaExpr, PitaFuncCallExpr, PitaFuncDefExpr, PitaIndexExpr, PitaLetExpr, PitaNumExpr, PitaSingleNumExpr, PitaVarExpr, gcd, pita_map, pita_num_map

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


def stage_conditional(expr: PitaNumExpr):
    match expr:
        case PitaCondExpr(left, right, true, false):
            try:
                left_val = evaluate_single_plaintext(left)
                right_val = evaluate_single_plaintext(right)
            except:
                return PitaCondExpr(left, right, stage_conditional(true), stage_conditional(false))
            if left_val < right_val:
                return stage_conditional(true)
            return stage_conditional(false)
        case _:
            return pita_num_map(expr, stage_conditional)


def evaluate_single_plaintext(expr: PitaSingleNumExpr) -> int:
    # this shoul be plaintext, beta reduction and subsitution should already be performed
    match expr:
        case PitaVarExpr(name):
            if name.isnumeric():
                return int(name)
            raise Exception(f'Unsubstituted variable `{name}`!')
        case PitaArithExpr(left, op, right):
            assert isinstance(left, PitaSingleNumExpr)
            assert isinstance(right, PitaSingleNumExpr)
            left_val = evaluate_single_plaintext(left)
            right_val = evaluate_single_plaintext(right)
            
            if op == '+':
                return left_val + right_val
            if op == '-':
                return left_val - right_val
            if op == '*':
                return left_val * right_val
            raise Exception(f'Unrecognized op `{op}`!')
        case PitaCondExpr(left, right, true, false):
            assert isinstance(true, PitaSingleNumExpr)
            assert isinstance(false, PitaSingleNumExpr)
            if evaluate_single_plaintext(left) < evaluate_single_plaintext(right):
                return evaluate_single_plaintext(true)
            return evaluate_single_plaintext(false)
        case _:
            raise TypeError(f'Found unexpected form `{expr}` when evaluating plaintext! (Maybe you forgot to do beta reduction?)')
                    
def desugar_indices(expr: PitaExpr):
    match expr:
        case PitaIndexExpr(arr, index):
            index = desugar_indices(index) # evaluate nested indices (e.g. a[b[i]])
            assert isinstance(index, PitaSingleNumExpr) # to satisfy the typechecker?
            return desugar_indices(arr.elems[evaluate_single_plaintext(index)])
        case _:
            return pita_map(expr, desugar_indices)


def substitute(var: str, val: PitaNumExpr, body: PitaNumExpr) -> PitaNumExpr:
    match body:
        case PitaVarExpr(name):
            if name == var:
                return val
            return body
        case _:
            return pita_num_map(body, lambda e: substitute(var, val, e))


def inline(func: str, args: list[str], body: PitaNumExpr, prog: PitaNumExpr):
    match prog:
        case PitaFuncCallExpr(name, params):
            params = [inline(func, args, body, param) for param in params]
            if name == func:
                evaluated_body = body
                for arg, param in zip(args, params):
                    evaluated_body = substitute(arg, param, evaluated_body)
                evaluated_body = stage_conditional(evaluated_body)
                return inline(func, args, body, evaluated_body)
            return PitaFuncCallExpr(name, params)
        case _:
            return pita_map(prog, lambda e: inline(func, args, body, e)) # type: ignore


def inline_all(expr: PitaExpr):
    match expr:
        case PitaLetExpr(var, val, body):
            match val:
                case PitaFuncDefExpr(args, func_body):
                    inlined_body = inline_all(body)
                    post_inlining = inline(var, args, func_body, inlined_body)
                    return post_inlining
                case _:
                    return PitaLetExpr(var, inline_all(val), inline_all(body))
        case _:
            return pita_map(expr, inline_all)

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


def check_plaintext(expr: PitaExpr, symbols=None):
    if symbols is None:
        symbols = {}
        is_ptxt(expr, symbols)
    
    # now, verify that all array index expressions are plaintext
    match expr:
        case PitaVarExpr(_):
            pass
        case PitaArithExpr(left, _, right):
            check_plaintext(left, symbols)
            check_plaintext(right, symbols)
        case PitaFuncDefExpr(_, body):
            check_plaintext(body, symbols)
        case PitaLetExpr(_, val, body):
            check_plaintext(val, symbols)
            check_plaintext(body, symbols)
        case PitaCondExpr(left, right, true, false):
            check_plaintext(left, symbols)
            check_plaintext(right, symbols)
            check_plaintext(true, symbols)
            check_plaintext(false, symbols)
        case PitaIndexExpr(arr, index):
            check_plaintext(arr, symbols)
            assert is_ptxt(index, symbols), f'{index} not plaintext!'
        case PitaFuncCallExpr(_, params):
            for param in params:
                check_plaintext(param, symbols)
        case PitaArrayExpr(elems):
            for elem in elems:
                check_plaintext(elem, symbols)
        case _:
            raise TypeError(type(expr))
    return expr


passes = [inline_all, substitute_all, check_plaintext, desugar_indices, stage_conditional, treeify_expr, comparison_folding, lambda t: t.normalize()]

def compile(prog) -> ChallahTree:
    for p in passes:
        prog = p(prog)
    return prog