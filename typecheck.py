from pita import (PitaArithExpr, PitaArrayExpr, PitaCondExpr, PitaExpr,
                  PitaFuncCallExpr, PitaFuncDefExpr, PitaIndexExpr,
                  PitaLetExpr, PitaVarExpr)


def is_ptxt(prog: PitaExpr, symbols: dict[str, bool]) -> bool:
    match prog:
        case PitaVarExpr(name):
            if name.isnumeric():
                return True # numeric literal
            if name in symbols:
                return symbols[name] # already bound this symbol
            return False # unbound symbol, must be an input wire
        case PitaArithExpr(left, _, right):
            return is_ptxt(left, symbols) and is_ptxt(right, symbols)
        case PitaArrayExpr(elems):
            return all([is_ptxt(elem, symbols) for elem in elems])
        case PitaLetExpr(name, val, body):
            symbols[name] = is_ptxt(val, symbols)
            return is_ptxt(body, symbols)
        case PitaCondExpr(left, lt, right, true, false):
            return is_ptxt(left, symbols) and is_ptxt(right, symbols) and is_ptxt(true, symbols) and is_ptxt(false, symbols)
        case PitaFuncDefExpr(_, _):
            return True # functions are polymorphic over Rep[*] and should be typechecked at the call-site; don't bother failing here
        case PitaIndexExpr(arr, index):
            return is_ptxt(arr, symbols) and is_ptxt(index, symbols)
        case PitaFuncCallExpr(_, params):
            return all([is_ptxt(param, symbols) for param in params])
        
        
def promote_op(a, b, op):
    if isinstance(a, list) and isinstance(b, list):
        return [op(i, j) for i, j in zip(a, b)]
    if isinstance(a, list):
        b = [b for _ in a]
        return promote_op(a, b, op)
    if isinstance(b, list):
        a = [a for _ in b]
        return promote_op(a, b, op)
    return op(a, b)
        
# def evaluate_ptxt(prog: PitaNumExpr, constants: dict[str, int | list[int]]) -> int | list[int]:
#     match prog:
#         case PitaVarExpr(name):
#             if name.isnumeric():
#                 return int(name)
#             if name in constants:
#                 return constants[name]
#             raise TypeError(f'{name} not in context!')
#         case PitaArithExpr(left, op, right):
#             left_val = evaluate_ptxt(left, constants)
#             right_val = evaluate_ptxt(right, constants)
#             if op == '+':
#                 return promote_op(left_val, right_val, lambda a, b: a + b)
#             if op == '-':
#                 return promote_op(left_val, right_val, lambda a, b: a - b)
#             if op == '*':
#                 return promote_op(left_val, right_val, lambda a, b: a * b)
#             raise TypeError(f'Op `{op}` not recognized')
#         case PitaArrayExpr(elems):
#             return [cast(int, evaluate_ptxt(elem, constants)) for elem in elems]
#         case PitaLetExpr(name, val, body):
#             return evaluate_ptxt(body, constants)
#         case PitaCondExpr(left, right, true, false):
#             left_val = cast(int, evaluate_ptxt(left, constants))
#             right_val = cast(int, evaluate_ptxt(right, constants))
#             if left_val < right_val:
#                 return evaluate_ptxt(true, constants)
#             return evaluate_ptxt(false, constants)
#         case PitaIndexExpr(arr, index):
#             return cast(list[int], evaluate_ptxt(arr, constants))[cast(int, evaluate_ptxt(index, constants))]
#         case PitaFuncCallExpr(_, params):
#             pass
        
        
            