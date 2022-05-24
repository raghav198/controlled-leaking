from __future__ import annotations
from typing import Dict, List, Set, Tuple, Union
from dataclasses import dataclass


Tree = Union['Leaf', 'Branch']

@dataclass
class Leaf:
    cost: int
    tag: int = -1

@dataclass
class Branch:

    left: Tree
    right: Tree
    factor: float
    cost: int
    tag: int = -1

    @property
    def p_left(self):
        return self.factor
    
    @property
    def p_right(self):
        return 1 - self.factor




class SliceCalculator:
    def __init__(self, tree: Tree):
        self.tree = tree
        self.tag_lookup: Dict[int, Tree] = {}
        self.cur_tag = 0
        self.children: Dict[int, Tuple[int]] = {}
        self.prob: Dict[int, float] = {}
        self.total_cost: Dict[int, int] = {}
        self.optimal_slice: Dict[int, Set[int]] = {}

        self.stack = []

    def push(self):
        self.stack.append((self.total_cost.copy(), self.optimal_slice.copy()))

    def pop(self):
        self.total_cost, self.optimal_slice = self.stack.pop()


    def tag(self):
        self.prob[0] = 1
        def traverse_up(tree: Tree):
            if isinstance(tree, Branch):
                # do the bottom first
                traverse_up(tree.left)
                traverse_up(tree.right)

                # cost
                self.total_cost[self.cur_tag] = self.total_cost[tree.left.tag] + self.total_cost[tree.right.tag] + tree.cost
            else:
                self.total_cost[self.cur_tag] = tree.cost
                
            # tagging
            tree.tag = self.cur_tag
            self.tag_lookup[self.cur_tag] = tree
            
            self.cur_tag += 1

        def traverse_down(tree: Tree):
            # assume that the tree is tagged already
            if isinstance(tree, Branch):
                self.prob[tree.left.tag] = self.prob[tree.tag] * tree.p_left
                self.prob[tree.right.tag] = self.prob[tree.tag] * tree.p_right

                traverse_down(tree.left)
                traverse_down(tree.right)


        traverse_up(self.tree) # tag and mark costs
        self.prob[self.tree.tag] = 1
        traverse_down(self.tree) # mark probabilities

    def enumerate_children(self):
        if not self.tag_lookup:
            self.tag()
        for key in self.tag_lookup:
            if isinstance(self.tag_lookup[key], Leaf):
                continue
            self.children[key] = (self.tag_lookup[key].left.tag, self.tag_lookup[key].right.tag)

    def compute_slice(self, tree_idx: int = 0) -> Set[int]:
        tree = self.tag_lookup[tree_idx]
        if not self.children:
            self.enumerate_children()
        if isinstance(tree, Leaf):
            self.optimal_slice[tree_idx] = {tree_idx}
            return {tree_idx}

        left_slice = self.compute_slice(tree.left.tag)
        right_slice = self.compute_slice(tree.right.tag)

        def check_lifts(slc: Set[int], inclusion: float, root: int):
            triples: List[int] = []
            for key in self.children:
                if set(self.children[key]).issubset(slc):
                    triples.append(key)

            while len(triples):
                stuff_added: Set[int] = set()
                for A in triples:
                    A_tree = self.tag_lookup[A]
                    assert isinstance(A_tree, Branch)

                    i = inclusion
                    p = self.prob[A] / self.prob[root]
                    x = A_tree.p_left

                    # A_cost = A_tree.cost
                    A_cost = self.total_cost[A]
                    B_cost = self.total_cost[A_tree.left.tag]
                    C_cost = self.total_cost[A_tree.right.tag]

                    # is it worth it to pull up?
                    thresh = A_tree.cost / (p * (A_cost - x * B_cost + (x - 1) * C_cost))

                    if len(self.stack):
                        old_cost, _ = self.stack[-1]
                        old_A = old_cost[A]
                        old_B = old_cost[A_tree.left.tag]
                        old_C = old_cost[A_tree.right.tag]

                        print(f'A: {old_A} -> {A_cost} (eA = {old_A - A_cost})')
                        print(f'B: {old_B} -> {B_cost} (eB = {old_B - B_cost})')
                        print(f'C: {old_C} -> {C_cost} (eC = {old_C - C_cost})')
                        print(f'Old: {old_A} = {A_tree.cost} + {old_B} + {old_C}')
                        print(f'New: {A_cost} vs {A_tree.cost + B_cost + C_cost}')
                        print(f'Additive failure: {A_tree.cost + B_cost + C_cost - A_cost}')

                        old_thresh = A_tree.cost / (p * (old_A - x * old_B + (x - 1) * old_C))
                        print(f'Threshold: {old_thresh} -> {thresh}')
                        input()

                    if i <= thresh:    
                        slc.remove(A_tree.left.tag)
                        slc.remove(A_tree.right.tag)
                        slc.add(A)
                        stuff_added.add(A)


                triples = []
                for key in self.children:
                    children = set(self.children[key])
                    if children.intersection(stuff_added) and children.issubset(slc):
                        triples.append(key)

            return slc

        print(f'Combined cost without lifting slices: {self.sliced_cost(tree_idx, left_slice | right_slice)}')

        left_lifted = check_lifts(left_slice, tree.p_left, tree.left.tag)
        right_lifted = check_lifts(right_slice, tree.p_right, tree.right.tag)

        print(f'Combined cost after lifting: {self.sliced_cost(tree_idx, left_lifted | right_lifted)}')
        input()

        self.optimal_slice[tree_idx] = left_lifted | right_lifted

        return left_lifted | right_lifted

    def sliced_cost(self, tree_idx: int, slc: Set[int]) -> float:
        cost = self.total_cost[tree_idx]
        for vertex in slc:
            cost -= self.total_cost[vertex] * (1 - self.prob[vertex])
        return cost

    def update_costs(self):
        self.push()
        for key in self.total_cost:
            self.total_cost[key] = self.sliced_cost(key, self.optimal_slice[key])
        self.optimal_slice = {}

leaves = [Leaf(2) for _ in range(6)]
D = Branch(leaves[0], leaves[1], factor=0.25, cost=1)
E = Branch(leaves[2], leaves[3], factor=0.25, cost=5)
C = Branch(leaves[4], leaves[5], factor=0.25, cost=10)
B = Branch(D, E, factor=0.25, cost=10)
A = Branch(B, C, factor=0.25, cost=1)


def complete_tree(depth):
    if depth > 0:
        return Branch(complete_tree(depth - 1), complete_tree(depth - 1), factor=0.5, cost=1)
    return Leaf(1)

T = complete_tree(6)

def compute_N_slices(T: Tree, N: int) -> List[Set[int]]:
    calc = SliceCalculator(T)
    calc.enumerate_children()

    for _ in range(N):
        calc.compute_slice(T.tag)
        calc.update_costs()

    slices = [{T.tag}]
    for _ in range(N):
        calc.pop()
        new_slice = set()
        for vertex in slices[-1]:
            new_slice.update(calc.optimal_slice[vertex])
        slices.append(new_slice)

    return slices[1:]

calc = SliceCalculator(T)
calc.enumerate_children()
print(calc.compute_slice(T.tag))
calc.update_costs()
print(calc.compute_slice(T.tag))

# print(list(map(len, compute_N_slices(T, 1))))
# print(list(map(len, compute_N_slices(T, 2))))
# print(list(map(len, compute_N_slices(T, 3))))