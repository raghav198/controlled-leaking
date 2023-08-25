from io import TextIOWrapper
import lark
import pita

class ASTTransformer(lark.Transformer):
    def var(self, name):
        return pita.PitaVarExpr(name[0].value)
    
    def ctxt(self, num):
        return pita.PitaVarExpr(f'input#{num[0].value}')
    
    def arith(self, items):
        return pita.PitaArithExpr(items[0], items[1], items[2])
    
    def let(self, items):
        return pita.PitaLetExpr(items[0].name, items[1], items[2])
    
    def cond(self, items):
        return pita.PitaCondExpr(items[0], items[1] == '<', items[2], items[3], items[4])
    
    def func_args(self, items):
        return list(items)
    
    def func_params(self, items):
        return list(items)
    
    def func_def(self, items):
        return pita.PitaFuncDefExpr(items[0], items[1])
    
    def func_call(self, items):
        return pita.PitaFuncCallExpr(items[0], items[1])
    
    def index_expr(self, items):
        return pita.PitaIndexExpr(items[0], items[1])
    
    def single_array_update(self, items):
        return tuple(items)
    
    def array_update(self, items):
        return pita.PitaUpdateExpr(items[0], items[1], items[2])
    
    def ctxt_array(self, items):
        return pita.PitaArrayExpr([pita.PitaVarExpr(f'input#{i}') for i in range(items[0], items[1])])
    
    def new_array(self, items):
        return pita.PitaArrayExpr([pita.PitaVarExpr('0') for _ in range(items[0])])
    
    def num_literal(self, items):
        return pita.PitaVarExpr(items[0].value)
    
        
    plus = lambda self, _: "+"
    minus = lambda self, _: "-"
    times = lambda self, _: "*"
    
    lt = lambda self, _: "<"
    eq = lambda self, _: "=="

    

def parse_string(string: str):
    parser = lark.Lark(open('grammar.lark').read())
    parse_tree = parser.parse(string)
    return ASTTransformer().transform(parse_tree)

def parse_file(file: TextIOWrapper):
    return parse_string(file.read())
    
    
if __name__ == '__main__':
    example = '''
let x = (1 + 3) in
if (x < 4) { 1 } else { 12 }
'''

    print(parse_string(example))