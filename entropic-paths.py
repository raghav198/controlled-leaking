from typing import Dict, Set
from math import log2
from coyote.coyote.coyote import *

class Branch:
    def __init__(self, cost: int = 1, probability: float = 0.5):
        self.cost = cost
        self.probability = probability

    def __repr__(self) -> str:
        return f'Branch({self.cost}, {self.probability:.2f})'


Tree = Dict[int, Branch]

def make_tree(index: int, depth: int, cur: Tree = {}):
    cur[index] = Branch()

    if depth == 0:
        return cur

    cur = make_tree(index * 2, depth - 1, cur)
    cur = make_tree(index * 2 + 1, depth - 1, cur)
    return cur


tree = make_tree(1, 2)

def visiting_probabilities(tree: Tree, node: int, acc: float, probabilities: Dict[int, float] = {}):
    probabilities[node] = acc
    if node * 2 in tree:
        visiting_probabilities(tree, node * 2, tree[node].probability * acc, probabilities)
    
    if node * 2 + 1 in tree:
        visiting_probabilities(tree, node * 2 + 1, (1 - tree[node].probability) * acc, probabilities)


def bits_leaked(tree: Tree, slice: Set[int], root: int = 1) -> float:
    probabilities: Dict[int, float] = {}
    visiting_probabilities(tree, root, 1, probabilities)

    print(probabilities)

    entropy = 0
    for node in slice:
        entropy -= probabilities[node] * log2(probabilities[node])

    return entropy


def tree_cost(tree: Tree, root: int) -> int:
    if root not in tree:
        return 0

    return tree[root].cost + tree_cost(tree, root * 2) + tree_cost(tree, root * 2 + 1)



def all_slices(tree: Tree, cur=set()):
    yield cur
    for val in cur:
        if val * 2 in tree and val * 2 + 1 in tree:
            yield from all_slices(tree, cur - {val} | {val * 2, val * 2 + 1})


def slicing_cost(tree: Tree, slice: Set[int]):
    probs: Dict[int, float] = {}
    def visit(node: int, acc: float = 1):
        if node not in tree:
            raise Exception('Invalid slice: incomplete')
        probs[node] = acc
        if node not in slice:
            visit(node * 2, acc * tree[node].probability)
            visit(node * 2 + 1, acc * (1 - tree[node].probability))

    visit(1)

    cost = 0

    entropy = 0
    for node in slice:
        if node not in probs:
            raise Exception(f'Invalid slice: ill-defined ({node})')
        
        entropy -= probs[node] * log2(probs[node])
        cost += probs[node] * tree_cost(tree, node)

    return entropy, cost

# for slice in all_slices(tree, {1}):
#     print(slice, slicing_cost(tree, slice))



gcd_tree = {}
gcd_tree[1] = Branch(0)
gcd_tree[2] = Branch(1)
gcd_tree[3] = Branch(1)
gcd_tree[4] = Branch(2)
gcd_tree[5] = Branch(3)
gcd_tree[6] = Branch(2)
gcd_tree[7] = Branch(3)

# TODO: slicing inlines the intermediates, so we need to be aware of that when computing preparation cost for each tree

for slice in all_slices(gcd_tree, {1}):
    print(slice, slicing_cost(gcd_tree, slice))