from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, TypeVar, cast

from numpy import interp


@dataclass
class PitaVarExpr:
    name: str

    def __repr__(self):
        return self.name

@dataclass
class PitaArithExpr:
    left: PitaNumExpr
    op: str
    right: PitaNumExpr
    
    def __repr__(self):
        return f'({self.left} {self.op} {self.right})'
        
        
@dataclass
class PitaArrayExpr:
    elems: list[PitaSingleNumExpr] # no array of arrays
    
    def __repr__(self):
        return repr(self.elems)
        
@dataclass
class PitaLetExpr:
    name: str
    val: PitaExpr
    body: PitaNumExpr
    
    def __repr__(self):
        return f'let {self.name} = {self.val} in\n{self.body}'
    
@dataclass
class PitaUpdateExpr:
    name: str
    updates: list[tuple[PitaNumExpr, PitaNumExpr]]
    body: PitaNumExpr
    
    def __repr__(self) -> str:
        updates = "\n\t".join(f"{idx} := {expr}" for idx, expr in self.updates)
        return f'update {self.name} {{\n\t{updates}\n}} in\n{self.body}'
    
    
@dataclass
class PitaCondExpr:
    left: PitaSingleNumExpr
    lt: bool
    right: PitaSingleNumExpr
    true: PitaNumExpr
    false: PitaNumExpr
    
    def __repr__(self):
        return f'(if ({self.left} {"<" if self.lt else "=="} {self.right}) {{ {self.true} }} else {{ {self.false} }})'
    
    
@dataclass
class PitaFuncDefExpr:
    args: list[str]
    body: PitaNumExpr
    recursive: bool = False
    
    def __repr__(self):
        return f'({", ".join(self.args)}) => {{ {self.body} }}'
    
    
@dataclass
class PitaIndexExpr:
    arr: PitaArrayExpr
    index: PitaSingleNumExpr
    
    def __repr__(self):
        return f'{self.arr}[{self.index}]'
    
    
@dataclass
class PitaFuncCallExpr:
    name: str
    params: list[PitaNumExpr]
    
    def __repr__(self):
        return f'{self.name}({", ".join([repr(param) for param in self.params])})'
    
    
def pita_map(prog: PitaExpr, f: Callable[[PitaExpr], PitaNumExpr]):
    match prog:
        case PitaLetExpr(var, val, body):
            return PitaLetExpr(var, f(val), f(body))
        case _:
            return pita_num_map(prog, f)
    
def pita_num_map(prog: PitaExpr, f: Callable[[PitaNumExpr], PitaNumExpr]):
    match prog:
        case PitaVarExpr(_):
            return prog
        case PitaArithExpr(left, op, right):
            return PitaArithExpr(f(left), op, f(right))
        case PitaLetExpr(var, val, body):
            if isinstance(val, PitaFuncDefExpr):
                return PitaLetExpr(var, val, f(body))
            return PitaLetExpr(var, f(val), f(body))
        case PitaCondExpr(left, lt, right, true, false):
            return PitaCondExpr(cast(PitaSingleNumExpr, f(left)), lt, cast(PitaSingleNumExpr, f(right)), f(true), f(false))
        case PitaIndexExpr(arr, index):
            # assert isinstance(farr := f(arr), PitaArrayExpr), 'Cannot reduce array in mapping'
            return PitaIndexExpr(f(arr), f(index)) # type: ignore
        case PitaFuncCallExpr(name, params):
            return PitaFuncCallExpr(name, [f(param) for param in params])
        case PitaArrayExpr(elems):
            return PitaArrayExpr(elems=[cast(PitaSingleNumExpr, f(elem)) for elem in elems])
    raise TypeError(type(prog))
    
PitaSingleNumExpr = PitaVarExpr | PitaArithExpr | PitaLetExpr | PitaCondExpr | PitaFuncCallExpr | PitaIndexExpr
PitaNumExpr = PitaSingleNumExpr | PitaArrayExpr
PitaExpr = PitaNumExpr | PitaFuncDefExpr

# let x = 5 in (let y = x * x in (x + y))
pita_prague = PitaLetExpr("x", PitaVarExpr("5"), PitaLetExpr("y", PitaArithExpr(PitaVarExpr("x"), "*", PitaVarExpr("x")), PitaArithExpr(PitaVarExpr("x"), "+", PitaVarExpr("y"))))


def interpret(expr: PitaExpr, vals: dict[str, int], funcs: dict[str, PitaFuncDefExpr]) -> int:
    match expr:
        case PitaVarExpr(name):
            try: return int(name)
            except: pass
            if name not in vals: raise ValueError("u suk")
            return vals[name]
        case PitaArithExpr(left, op, right):
            left_eval = interpret(left, vals, funcs)
            right_eval = interpret(right, vals, funcs)
            if op in ["+", "-", "*"]: return int(eval(str(left_eval) + op + str(right_eval)))
            raise SystemExit("..|..")
        case PitaLetExpr(name, val, body):
            if isinstance(val, PitaFuncDefExpr): funcs[name] = val
            elif isinstance(val, PitaNumExpr): vals[name] = interpret(val, vals, funcs)
            return interpret(body, vals, funcs)
        case PitaCondExpr(left, lt, right, true, false):
            if lt:
                if interpret(left, vals, funcs) < interpret(right, vals, funcs): return interpret(true, vals, funcs)
                return interpret(false, vals, funcs)
            if interpret(left, vals, funcs) == interpret(right, vals, funcs): return interpret(true, vals, funcs)
            return interpret(false, vals, funcs)
        case PitaFuncCallExpr(name, params):
            if name not in funcs: raise RecursionError("lmao get destroyed nerdbag")
            func = funcs[name]
            local_context = vals.copy()
            if len(params) != len(func.args): raise IOError('ur dump')
            for param, arg in zip(params, func.args): local_context[arg] = interpret(param, vals.copy(), funcs.copy())
            return interpret(func.body, local_context, funcs)
        case _:
            raise OSError("bout 5 pounds")
    


"""
a, b => (18, 12)

let max = ((x, y) => {if (x < y) {y} else {x}}) in
let min = ((x, y) => {if (x < y) {x} else {y}}) in
let c = min(a, b) in
let d = max(a, b) - c in
max(c, d) - min(c, d)
"""

def _gcd(a, b, n):
    if n == 1:
        return PitaArithExpr(PitaFuncCallExpr('max', [a, b]), '-', PitaFuncCallExpr('min', [a, b]))
    return PitaLetExpr(f't{n}_1', PitaFuncCallExpr('min', [a, b]), 
                       PitaLetExpr(f't{n}_2', PitaArithExpr(PitaFuncCallExpr('max', [a, b]), '-', PitaVarExpr(f't{n}_1')), 
                                   _gcd(PitaVarExpr(f't{n}_1'), PitaVarExpr(f't{n}_2'), n - 1)))
    
def gcd(a, b, n):
    return PitaLetExpr('max', PitaFuncDefExpr(['x', 'y'], PitaCondExpr(PitaVarExpr('x'), True, PitaVarExpr('y'), PitaVarExpr('y'), PitaVarExpr('x'))),
                              PitaLetExpr('min', PitaFuncDefExpr(['x', 'y'], PitaCondExpr(PitaVarExpr('x'), True, PitaVarExpr('y'), PitaVarExpr('x'), PitaVarExpr('y'))),
                                          _gcd(a, b, n)))

pita_prague_too = PitaLetExpr('max', PitaFuncDefExpr(['x', 'y'], PitaCondExpr(PitaVarExpr('x'), True, PitaVarExpr('y'), PitaVarExpr('y'), PitaVarExpr('x'))),
                              PitaLetExpr('min', PitaFuncDefExpr(['x', 'y'], PitaCondExpr(PitaVarExpr('x'), True, PitaVarExpr('y'), PitaVarExpr('x'), PitaVarExpr('y'))),
                                          PitaLetExpr('c', PitaFuncCallExpr('min', [PitaVarExpr('a'), PitaVarExpr('b')]), 
                                                      PitaLetExpr('d', PitaArithExpr(PitaFuncCallExpr('max', [PitaVarExpr('a'), PitaVarExpr('b')]), '-', PitaVarExpr('c')),
                                                                  PitaArithExpr(PitaFuncCallExpr('max', [PitaVarExpr('c'), PitaVarExpr('d')]), '-', PitaFuncCallExpr('min', [PitaVarExpr('c'), PitaVarExpr('d')]))))))


if __name__ == '__main__':
    print(interpret(pita_prague, {}, {}))
    print(interpret(pita_prague_too, {'a': 18, 'b': 12}, {}))
