from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Callable, TypeAlias

from pita import (
    PitaArithExpr,
    PitaArrayExpr,
    PitaCondExpr,
    PitaExpr,
    PitaFuncCallExpr,
    PitaFuncDefExpr,
    PitaIndexExpr,
    PitaLetExpr,
    PitaNumExpr,
    PitaSingleNumExpr,
    PitaUpdateExpr,
    PitaVarExpr,
    pita_map,
    pita_num_map,
)

from eq_neq import EqNeq


class PitaException(Exception):
    """Wrapper for all PITA interpreter exceptions"""


def assert_compatible(*_):
    """Ensure that `left` and `right` are 'compatible' as arguments to a function (e.g. if they are arrays, they have the same size)"""
    return True  # TODO: bruh


@dataclass(frozen=True)
class ChallahVar:
    """Leaf node representing a variable"""

    name: str

    def __repr__(self):
        return self.name

    def normalize(self):
        """Normalize Challah tree by pushing all arrays to leaves"""
        return self


@dataclass(frozen=True)
class ChallahArithExpr:
    """Leaf node representing an arithmetic expression"""

    # check sizes
    def __post_init__(self):
        # assert isinstance(self.left, ChallahArray) == isinstance(self.right, ChallahArray)
        # if isinstance(self.left, ChallahArray) and isinstance(self.right, ChallahArray):
        #     assert len(self.left.elems) == len(self.right.elems)
        assert_compatible(self.left, self.right)

    left: ChallahLeaf
    op: str
    right: ChallahLeaf

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"

    def normalize(self) -> ChallahArray | ChallahArithExpr:
        """Normalize Challah tree by pushing all arrays to leaves"""
        if isinstance(self.left, ChallahArray) and isinstance(self.right, ChallahArray):
            return ChallahArray(
                elems=[
                    ChallahArithExpr(left.normalize(), self.op, right.normalize()) for left, right in zip(self.left.elems, self.right.elems)
                ]
            )
        return self


@dataclass(frozen=True)
class ChallahArray:
    """Leaf node representing an array of Challah expressions"""

    elems: list[ChallahVar | ChallahArithExpr]

    def __repr__(self):
        return repr(self.elems)

    def normalize(self):
        """Normalize Challah tree by pushing all arrays to leaves"""
        return self


@dataclass
class ChallahBranch:
    """Branch node representing a conditional statement"""

    left: ChallahLeaf
    lt: bool
    right: ChallahLeaf
    true: ChallahTree
    false: ChallahTree

    def __post_init__(self):
        assert_compatible(self.left, self.right)
        assert_compatible(self.true, self.false)

    def __repr__(self):
        return f'if ({self.left} {"<" if self.lt else "=="} {self.right}) {{\n{self.true}\n}} else {{\n{self.false}\n}}'

    def normalize(self):
        """Normalize Challah tree by pushing all arrays to leaves"""
        true = [t.normalize() for t in self.true] if isinstance(self.true, list) else self.true.normalize()
        false = [t.normalize() for t in self.false] if isinstance(self.false, list) else self.false.normalize()

        return ChallahBranch(self.left, self.lt, self.right, true, false)


ChallahLeaf: TypeAlias = ChallahVar | ChallahArithExpr | ChallahArray
ChallahTree: TypeAlias = ChallahLeaf | list[ChallahLeaf] | ChallahBranch


def pprint(tree: ChallahTree, depth: int = 0, start=""):
    """Pretty-print a Challah tree"""
    if isinstance(tree, ChallahBranch):
        print(f'{" " * depth * 4}{start}{tree.left} {"<" if tree.lt else "=="} {tree.right}')
        pprint(tree.true, depth + 1, start="\u2713 ")
        pprint(tree.false, depth + 1, start="\u2717 ")
    else:
        print(" " * depth * 4 + start + str(tree))


def operad_compose(left: ChallahTree, right: ChallahTree, f: Callable[[ChallahLeaf, ChallahLeaf], ChallahTree]):
    """Compose two decision trees according to a reduction function `f`

    Args:
        left (ChallahTree): Tree of left arguments to `f`
        right (ChallahTree): Tree of right arguments to `f`
        f (Callable[[ChallahLeaf, ChallahLeaf], ChallahTree]): Reduction to apply at every leaf

    Raises:
        TypeError: if `left` and `right` are not both trees

    Returns:
        ChallahTree: Tensor product of `left` and `right`, with `f` applied at every leaf
    """
    if isinstance(left, ChallahLeaf) and isinstance(right, ChallahLeaf):
        return f(left, right)
    if isinstance(left, ChallahBranch):
        return ChallahBranch(left.left, left.lt, left.right, operad_compose(left.true, right, f), operad_compose(left.false, right, f))
    if isinstance(right, ChallahBranch):
        return ChallahBranch(right.left, right.lt, right.right, operad_compose(left, right.true, f), operad_compose(left, right.false, f))
    raise TypeError(type(left), type(right))


def combine_trees(left: ChallahTree, op: str, right: ChallahTree):
    """Combine two trees with an arithmetic operator"""
    return operad_compose(left, right, lambda lhs, rhs: ChallahArithExpr(lhs, op, rhs))


def compare_trees(left: ChallahTree, lt: bool, right: ChallahTree, true: ChallahTree, false: ChallahTree):
    """Combine two trees with a comparison operator"""
    return operad_compose(left, right, lambda lhs, rhs: ChallahBranch(lhs, lt, rhs, true, false))


def concat(left: ChallahLeaf, right: ChallahLeaf) -> ChallahArray:
    """Concatenate two leaves into a single array"""

    def arr(leaf: ChallahLeaf) -> ChallahArray:
        match leaf:
            case ChallahArray(_):
                return leaf
            case ChallahVar(_) | ChallahArithExpr(_, _, _):
                return ChallahArray(elems=[leaf])

    return ChallahArray(elems=arr(left).elems + arr(right).elems)


def concatenate_trees(left: ChallahTree, right: ChallahTree):
    """Combine an array of trees into a tree of arrays"""
    return operad_compose(left, right, concat)


def stage_conditional(expr: PitaNumExpr):
    """Partially evaluate all plaintext conditions in `expr`

    Args:
        expr (PitaNumExpr): Expression to partially evaluate

    Returns:
        PitaNumExpr: `expr` with all plaintext conditions evaluated
    """
    match expr:
        case PitaCondExpr(left, op, right, true, false):
            try:
                left_val = evaluate_single_plaintext(left)
                right_val = evaluate_single_plaintext(right)
            except PitaException:
                return PitaCondExpr(left, op, right, stage_conditional(true), stage_conditional(false))
            if left_val < right_val:
                return stage_conditional(true)
            return stage_conditional(false)
        case _:
            return pita_num_map(expr, stage_conditional)


def evaluate_single_plaintext(expr: PitaSingleNumExpr) -> int:
    """Interprets a plaintext expression into an integer

    Args:
        expr (PitaSingleNumExpr): Plaintext expression

    Raises:
        PitaException: if `expr` contains an unsubstituted variable
        PitaException: if `expr` contains an unrecognized operation
        TypeError: if `expr` is not one of `PitaVarExpr`, `PitaArithExpr`, or `PitaCondExpr`

    Returns:
        int: result of evaluating `expr`
    """
    # this should be plaintext, beta reduction and subsitution should already be performed
    match expr:
        case PitaVarExpr(name):
            if name.isnumeric():
                return int(name)
            raise PitaException(f"Unsubstituted variable `{name}`!")
        case PitaArithExpr(left, op, right):
            assert isinstance(left, PitaSingleNumExpr)
            assert isinstance(right, PitaSingleNumExpr)
            left_val = evaluate_single_plaintext(left)
            right_val = evaluate_single_plaintext(right)

            if op == "+":
                return left_val + right_val
            if op == "-":
                return left_val - right_val
            if op == "*":
                return left_val * right_val
            raise PitaException(f"Unrecognized op `{op}`!")
        case PitaCondExpr(left, op, right, true, false):
            assert isinstance(true, PitaSingleNumExpr)
            assert isinstance(false, PitaSingleNumExpr)
            if (op and evaluate_single_plaintext(left) < evaluate_single_plaintext(right)) or (
                not op and evaluate_single_plaintext(left) == evaluate_single_plaintext(right)
            ):
                return evaluate_single_plaintext(true)
            return evaluate_single_plaintext(false)
        case _:
            raise TypeError(f"Found unexpected form `{expr}` when evaluating plaintext! (Maybe you forgot to do beta reduction?)")


def desugar_indices(expr: PitaExpr):
    """Applies plaintext array indices"""
    match expr:
        case PitaIndexExpr(arr, index):
            index = desugar_indices(index)  # evaluate nested indices (e.g. a[b[i]])
            assert isinstance(index, PitaSingleNumExpr)  # to satisfy the typechecker?
            return desugar_indices(arr.elems[evaluate_single_plaintext(index)])
        case _:
            return pita_map(expr, desugar_indices)


def substitute(var: str, val: PitaNumExpr, body: PitaNumExpr) -> PitaNumExpr:
    """Replace every instance of `PitaVarExpr(var)` in `body` with `val`"""
    match body:
        case PitaVarExpr(name):
            if name == var:
                return val
            return body
        case _:
            return pita_num_map(body, lambda e: substitute(var, val, e))


def inline(func: str, args: list[str], body: PitaNumExpr, prog: PitaNumExpr):
    """Inline every call to `func` in `prog`

    Args:
        func (str): Function name
        args (list[str]): Formal arguments for `func`
        body (PitaNumExpr): Body of `func`
        prog (PitaNumExpr): Program in which to inline calls to `func`

    Returns:
        PitaNumExpr: `prog` with calls to `func` inlined
    """
    match prog:
        case PitaFuncCallExpr(name, params):
            params = [inline(func, args, body, param) for param in params]
            if name == func:
                evaluated_body = body
                for i, arg in enumerate(args):
                    evaluated_body = substitute(arg, PitaVarExpr(f"{func}#{i}"), evaluated_body)
                for i, param in enumerate(params):
                    evaluated_body = substitute(f"{func}#{i}", param, evaluated_body)
                # for arg, param in zip(args, params):
                #     evaluated_body = substitute(arg, param, evaluated_body)

                # for p in interpret_passes:
                #     evaluated_body = p(evaluated_body)

                return inline(func, args, body, evaluated_body)
            return PitaFuncCallExpr(name, params)
        case _:
            return pita_map(prog, lambda e: inline(func, args, body, e))  # type: ignore


def beta_reduce(
    args: list[str], body: PitaNumExpr, params: list[PitaNumExpr], ctx_vars: dict[str, PitaNumExpr], ctx_funcs: dict[str, PitaFuncDefExpr]
):
    """Perform beta-reduction by substituting parameters into a function body"""
    subst_vars: dict[str, PitaNumExpr] = ctx_vars.copy()
    # print(ctx_vars)
    # print(args)
    # print(params)
    for arg, param in zip(args, params):
        subst_vars[arg] = param
    # print(subst_vars)
    # input()
    return interpret(body, subst_vars, ctx_funcs)


# def inline_all(expr: PitaExpr):
#     match expr:
#         case PitaLetExpr(var, val, body):
#             match val:
#                 case PitaFuncDefExpr(args, func_body):
#                     inlined_body = inline_all(body)
#                     post_inlining = inline(var, args, func_body, inlined_body)
#                     return post_inlining
#                 case _:
#                     return PitaLetExpr(var, inline_all(val), inline_all(body))
#         case _:
#             return pita_map(expr, inline_all)


# def substitute_all(expr: PitaExpr):
#     match expr:
#         case PitaLetExpr(var, val, body):
#             assert isinstance(val, PitaNumExpr), f"Function definition {var} not inlined before substitution!"
#             return substitute(var, val, substitute_all(body))
#         case _:
#             return pita_num_map(expr, substitute_all)


def comparison_folding(tree: ChallahTree, known: list[tuple[ChallahLeaf, ChallahLeaf]], equalities: EqNeq[ChallahLeaf]):
    """Prune paths based on path conditions

    Args:
        tree (ChallahTree): Tree to prune
        known (list[tuple[ChallahLeaf, ChallahLeaf]], optional): List of known partial order facts. Defaults to [].
        equalities (EqNeq[ChallahLeaf], optional): Known equality/inequality facts. Defaults to EqNeq().

    Returns:
        ChallahTree: Pruned tree
    """

    # TODO: correctly handle numbers
    match tree:
        case ChallahBranch(left, lt, right, true, false):
            if lt:
                if (equalities.contains(left) and equalities.contains(right) and equalities.is_equivalent(left, right)) or left == right:
                    return comparison_folding(false, known, equalities)
                if (left, right) in known:
                    # print(f'Getting rid of {false} in {tree}: {left, right} in {known}')
                    return comparison_folding(true, known, equalities)
                if (right, left) in known:
                    # print(f'Getting rid of {true} in {tree}: {right, left} in {known}')
                    return comparison_folding(false, known, equalities)
                return ChallahBranch(
                    left,
                    lt,
                    right,
                    comparison_folding(true, known + [(left, right)], equalities),
                    comparison_folding(false, known + [(right, left)], equalities),
                )
            else:
                if equalities.contains(left) and equalities.contains(right):
                    if equalities.is_unequal(left, right):
                        # print(f'Getting rid of {true} in {tree}: {equalities.all_classes(), equalities.uneq}')
                        return comparison_folding(false, known, equalities)
                    if equalities.is_equivalent(left, right):
                        # print(f'Getting rid of {false} in {tree}: {equalities.all_classes(), equalities.uneq}')
                        return comparison_folding(true, known, equalities)

                true_branch = equalities.copy()
                if not true_branch.contains(left):
                    true_branch.add(left)
                if not true_branch.contains(right):
                    true_branch.add(right)

                false_branch = true_branch.copy()
                false_branch.require_neq(left, right)
                true_branch.union(left, right)
                return ChallahBranch(
                    left, lt, right, comparison_folding(true, known, true_branch), comparison_folding(false, known, false_branch)
                )
        case _:
            # print(f'=> {tree} (known {list(equalities.all_classes())})')
            return tree


def treeify_expr(expr: PitaExpr):
    """Convert `expr` into a decision tree by normalizing and pruning

    Args:
        expr (PitaExpr): Interpreted expression to normalize

    Raises:
        TypeError: if `expr` is not one of `PitaVarExpr`, `PitaArithExpr`, `PitaCondExpr`, or `PitaArrayExpr`

    Returns:
        ChallahTree: Treeified expression
    """
    match expr:
        case PitaVarExpr(name):
            return ChallahVar(name)
        case PitaArithExpr(left, op, right):
            ltree = treeify_expr(left)
            rtree = treeify_expr(right)

            return comparison_folding(combine_trees(ltree, op, rtree), known=[], equalities=EqNeq())

        case PitaCondExpr(left, lt, right, true, false):
            return comparison_folding(
                compare_trees(treeify_expr(left), lt, treeify_expr(right), treeify_expr(true), treeify_expr(false)),
                known=[],
                equalities=EqNeq(),
            )
        case PitaArrayExpr(elems):
            tree_elems = [treeify_expr(elem) for elem in elems]
            return comparison_folding(reduce(concatenate_trees, tree_elems[1:], tree_elems[0]), known=[], equalities=EqNeq())
    raise TypeError(type(expr))


# def check_plaintext(expr: PitaExpr, symbols=None):
#     if symbols is None:
#         symbols = {}
#         is_ptxt(expr, symbols)

#     # now, verify that all array index expressions are plaintext
#     match expr:
#         case PitaVarExpr(_):
#             pass
#         case PitaArithExpr(left, _, right):
#             check_plaintext(left, symbols)
#             check_plaintext(right, symbols)
#         case PitaFuncDefExpr(_, body):
#             check_plaintext(body, symbols)
#         case PitaLetExpr(_, val, body):
#             check_plaintext(val, symbols)
#             check_plaintext(body, symbols)
#         case PitaCondExpr(left, _, right, true, false):
#             check_plaintext(left, symbols)
#             check_plaintext(right, symbols)
#             check_plaintext(true, symbols)
#             check_plaintext(false, symbols)
#         case PitaIndexExpr(arr, index):
#             check_plaintext(arr, symbols)
#             assert is_ptxt(index, symbols), f"{index} not plaintext!"
#         case PitaFuncCallExpr(_, params):
#             for param in params:
#                 check_plaintext(param, symbols)
#         case PitaArrayExpr(elems):
#             for elem in elems:
#                 check_plaintext(elem, symbols)
#         case _:
#             raise TypeError(type(expr))
#     return expr


# def evaluate(prog: PitaNumExpr):
#     try:
#         match prog:
#             case PitaArrayExpr(elems):
#                 return PitaArrayExpr([cast(PitaSingleNumExpr, PitaVarExpr(str(evaluate_single_plaintext(elem)))) for elem in elems])
#             case _:
#                 return PitaVarExpr(str(evaluate_single_plaintext(prog)))
#     except PitaException:
#         return prog


ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}


def interpret(expr: PitaExpr, ctx_vars: dict[str, PitaNumExpr] = {}, ctx_funcs: dict[str, PitaFuncDefExpr] = {}) -> PitaNumExpr:
    """Run the PITA interpreter on `expr`

    Args:
        expr (PitaExpr): Expression to interpret
        ctx_vars (dict[str, PitaNumExpr], optional): Mapping of all variables defined in the current context. Defaults to {}.
        ctx_funcs (dict[str, PitaFuncDefExpr], optional): Mapping of all functions defined in the current context. Defaults to {}.

    Returns:
        PitaNumExpr: Result of interpreting `expr`
    """

    def available(e):
        return isinstance(e, PitaVarExpr) and e.name.isnumeric()

    def get(e):
        return int(e.name)

    match expr:
        case PitaVarExpr(name):
            if name in ctx_vars:
                # print(f'Looking up {name} = {ctx_vars[name]}')
                return interpret(ctx_vars[name], ctx_vars, ctx_funcs)
            if not name.isnumeric() and not name.startswith("input#"):
                raise PitaException(f"Undefined variable: `{name}`")
            return expr
        case PitaArithExpr(left, op, right):
            left_val = interpret(left, ctx_vars, ctx_funcs)
            right_val = interpret(right, ctx_vars, ctx_funcs)
            if available(left_val) and available(right_val):
                return PitaVarExpr(str(ops[op](get(left_val), get(right_val))))
            return PitaArithExpr(left_val, op, right_val)
        case PitaLetExpr(var, val, body):
            if isinstance(val, PitaNumExpr):
                ctx_vars[var] = val
                return interpret(body, ctx_vars, ctx_funcs)
            ctx_funcs[var] = val
            return interpret(body, ctx_vars, ctx_funcs)
        case PitaCondExpr(left, lt, right, true, false):
            # print(f'condition {left} {"<" if lt else "=="} {right}')
            left_val = interpret(left, ctx_vars, ctx_funcs)
            right_val = interpret(right, ctx_vars, ctx_funcs)
            # print(f'new condition {left_val} {"<" if lt else "=="} {right_val}')
            assert isinstance(left_val, PitaSingleNumExpr)
            assert isinstance(right_val, PitaSingleNumExpr)
            if available(left_val) and available(right_val):
                if (lt and get(left_val) < get(right_val)) or (not lt and get(left_val) == get(right_val)):
                    return interpret(true, ctx_vars, ctx_funcs)
                return interpret(false, ctx_vars, ctx_funcs)
            # print(f'new condition {left_val} {"<" if lt else "=="} {right_val}')
            return PitaCondExpr(left_val, lt, right_val, interpret(true, ctx_vars, ctx_funcs), interpret(false, ctx_vars, ctx_funcs))
        case PitaIndexExpr(arr, index):
            # print(f'index {arr}[{index}] ({ctx_vars})')
            arr_val = interpret(arr, ctx_vars, ctx_funcs)
            index_val = interpret(index, ctx_vars, ctx_funcs)
            assert isinstance(arr_val, PitaArrayExpr), arr_val
            assert available(index_val), "Array indices must be known at compile-time!"
            return arr_val.elems[get(index_val)]
        case PitaFuncCallExpr(name, params):
            assert name in ctx_funcs
            func = ctx_funcs[name]
            assert len(func.args) == len(
                params
            ), f"Function `{name}` called with the wrong number of parameters (expected {len(func.args)}, got {len(params)})"
            params = [interpret(param, ctx_vars, ctx_funcs) for param in params]
            # print(f'trace: {name}{tuple(params)}')
            try:
                return beta_reduce(func.args, func.body, params, ctx_vars, ctx_funcs)
            except RecursionError:
                assert False, f"Could not prove that function `{name}` terminates with a ciphertext argument!"
        case PitaArrayExpr(elems):
            # doing this for mypy
            new_elems: list[PitaSingleNumExpr] = []
            for elem in elems:
                new_elem = interpret(elem, ctx_vars, ctx_funcs)
                assert isinstance(new_elem, PitaSingleNumExpr)
                new_elems.append(new_elem)
            return PitaArrayExpr(new_elems)
        case PitaUpdateExpr(name, updates, body):
            indices = [interpret(idx, ctx_vars, ctx_funcs) for idx, _ in updates]
            exprs = [interpret(expr, ctx_vars, ctx_funcs) for _, expr in updates]

            assert all(available(idx) for idx in indices), "Array indices must be known at compile-time!"

            assert name in ctx_vars, f"Array `{name}` undefined!"
            to_update = ctx_vars[name]
            assert isinstance(to_update, PitaArrayExpr), f"Cannot update `{name}` as an array!"
            for idx, expr in zip(indices, exprs):
                assert isinstance(expr, PitaSingleNumExpr), "Arrays cannot be nested!"
                to_update.elems[get(idx)] = expr
            return interpret(body, ctx_vars, ctx_funcs)

    raise TypeError(type(expr))


# comparison_folding = lambda x: x
# interpret_passes = [inline_all, substitute_all, check_plaintext, desugar_indices, stage_conditional]
passes = [interpret, treeify_expr, lambda t: comparison_folding(t, known=[], equalities=EqNeq()), lambda t: t.normalize()]


def pita_compile(prog) -> ChallahTree:
    """Compile a PITA program into a decision tree by doing the following:
    1. Run the PITA interpreter to stage away plaintext computation
    2. Normalize the ciphertext expression into a decision tree
    3. Prune away impossible paths
    4. Normalize arrays of trees into a tree of arrays

    Args:
        prog (PitaExpr): PITA program to compile

    Returns:
        ChallahTree: Pruned decision tree encoding the semantics of `prog`
    """
    # print(prog)
    for p in passes:
        try:
            prog = p(prog)
        except AssertionError as e:
            print(e)
            quit()
    return prog
