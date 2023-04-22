import pyparsing as pp
import pita
import holla

var = pp.Word(pp.alphanums)
single_num_expr = pp.Forward()
num_expr = pp.Forward()
expr = pp.Forward()
func_def = pp.Forward()
op = pp.Literal('+') | '-' | '*'
literal_num = pp.Word(pp.nums)
arith = '(' + num_expr + op + num_expr + ')'
let = pp.Literal('let') + var + '=' + expr + 'in' + num_expr
cond = pp.Literal('if') + '(' + num_expr + '<' + num_expr + ')' + '{' + num_expr + '}' + 'else' + '{' + num_expr + '}'
func_def <<= pp.Literal('\\') + '(' + pp.Group(pp.delimited_list(var, ',')) + ')' + '=>' + '{' + num_expr + '}'

func_call = var + '(' + pp.Group(pp.delimited_list(num_expr, ',')) + ')'
index_expr = (pp.Suppress('@') + num_expr + '[' + single_num_expr + ']')

single_num_expr <<= cond | let | arith | func_call | var | index_expr | literal_num
num_expr <<= single_num_expr | (pp.Suppress('[') + pp.Group(pp.delimited_list(single_num_expr, ';')) + pp.Suppress(']')).set_parse_action(lambda n: pita.PitaArrayExpr(list(n[0])))
expr <<= func_def | num_expr

def IndexAction(n):
    return pita.PitaIndexExpr(n[0], n[3])

def VarAction(n):
    return pita.PitaVarExpr(n[0])

def ArithAction(n):
    return pita.PitaArithExpr(n[1], n[2], n[3])

def LetAction(n):
    return pita.PitaLetExpr(n[1].name, n[3], n[5])

def CondAction(n):
    return pita.PitaCondExpr(n[2], n[4], n[7], n[11])

def FuncDefAction(n):
    return pita.PitaFuncDefExpr(list(map(str, n[2])), n[6])

def FuncCallAction(n):
    return pita.PitaFuncCallExpr(n[0].name, list(n[2]))

var.set_parse_action(VarAction)
index_expr.set_parse_action(IndexAction)
literal_num.set_parse_action(VarAction)
let.set_parse_action(LetAction)
arith.set_parse_action(ArithAction)
cond.set_parse_action(CondAction)
func_def.set_parse_action(FuncDefAction)
func_call.set_parse_action(FuncCallAction)
if __name__ == '__main__':
    from sys import argv
    # print(open(argv[1]).read())
    # argv.append('prog_to_tree/gcd.pita')
    argv.append('pita_examples/bubble.pita')
    program = expr.parse_string(open(argv[1]).read(), parse_all=True)[0]
    # print(program)
    holla.pprint(holla.compile(program))

# ans = expr.parse_string('let min = \\(x, y) => { if (x < y) { x } else { y } } in min(a, b)', parse_all=True)
# print(holla.compile(ans[0]))