from math import log2
from slicing_dp import SliceCalculator, Tree, entropy

# T = complete_tree(3)

def entropy_slice(T: Tree, limit: float) -> tuple[set[int], SliceCalculator]:
    calc = SliceCalculator(T)
    calc.enumerate_children()
    # print(calc.children)
    slc = {T.tag}
    cur = entropy(T, slc)
    
    info = lambda n: float(-calc.prob[n] * log2(calc.prob[n]))
    
    def advance(slc: set[int], current_entropy: float) -> int | None:
        available = {n for n in slc
            if n in calc.children and sum(map(info, calc.children[n])) - info(n) + current_entropy <= limit}
        
        heuristic = lambda n: info(n) / sum(calc.prob[c] / calc.prob[n] * calc.total_cost[c] for c in calc.children[n])
        if len(available):
            return min(available, key=heuristic)
        return None
    
    while (node_to_advance := advance(slc, cur)):
        children = calc.children[node_to_advance]
        slc.remove(node_to_advance)
        slc.update(children)
        cur = entropy(T, slc)
    
    return slc, calc

# print(T)

# print(slc := entropy_slice(T, 1.8))
# print(entropy(T, slc[0]))