from __future__ import annotations
from typing import Dict, List, Set, Tuple, Union
from dataclasses import dataclass

from math import log2

from holla import ChallahTree


Tree = Union['Leaf', 'Branch']

@dataclass
class Leaf:
    cost: float
    tag: int = -1
    backref: ChallahTree | None = None
@dataclass
class Branch:

    left: Tree
    right: Tree
    factor: float
    cost: float
    tag: int = -1
    backref: ChallahTree | None = None

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
        self.children: Dict[int, Tuple[int, ...]] = {}
        self.prob: Dict[int, float] = {}
        self.total_cost: Dict[int, float] = {}
        self.original_total_cost: Dict[int, float] = {}
        self.optimal_slice: Dict[int, Set[int]] = {}

        self.stack: List[Tuple[Dict[int, float], Dict[int, Set[int]]]] = []

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
        self.original_total_cost = self.total_cost.copy()

    def enumerate_children(self):
        if not self.tag_lookup:
            self.tag()
        for key in self.tag_lookup:
            if isinstance(tree := self.tag_lookup[key], Leaf):
                continue
            self.children[key] = (tree.left.tag, tree.right.tag)

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

                    # if len(self.stack):
                    #     old_cost, _ = self.stack[-1]
                    #     old_A = old_cost[A]
                    #     old_B = old_cost[A_tree.left.tag]
                    #     old_C = old_cost[A_tree.right.tag]

                        # print(f'A = {A}, B = {A_tree.left.tag}, C = {A_tree.right.tag}...')

                        # print(f'Total A cost if we did not lift (actual should be lower): {A_tree.cost + B_cost + C_cost}')
                        # print(f'Actual total A cost: {A_cost}')

                        # print(f'A: {old_A} -> {A_cost} (eA = {old_A - A_cost})')
                        # print(f'B: {old_B} -> {B_cost} (eB = {old_B - B_cost})')
                        # print(f'C: {old_C} -> {C_cost} (eC = {old_C - C_cost})')
                        # print(f'Old: {old_A} = {A_tree.cost} + {old_B} + {old_C}')
                        # print(f'New: {A_cost} vs {A_tree.cost + B_cost + C_cost}')
                        # print(f'Additive failure: {A_tree.cost + B_cost + C_cost - A_cost}')

                        # old_thresh = A_tree.cost / (p * (old_A - x * old_B + (x - 1) * old_C))
                        # print(f'Threshold: {old_thresh} -> {thresh}')
                        # input()
                        # assert thresh >= old_thresh, f'{A_tree.left.tag, A_tree.right.tag} -> {A}'

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



        left_lifted = check_lifts(left_slice, tree.p_left, tree.left.tag)
        right_lifted = check_lifts(right_slice, tree.p_right, tree.right.tag)

        self.optimal_slice[tree_idx] = left_lifted | right_lifted

        # assert self.sliced_cost(tree_idx, left_slice | right_slice) >= self.sliced_cost(tree_idx, left_lifted | right_lifted), f'{tree.left.tag} <- {tree_idx} -> {tree.right.tag}'

        return left_lifted | right_lifted

    def sliced_cost(self, tree_idx: int, slc: Set[int]) -> float:
        cost = 0.
        for vertex in slc:
            cost += self.total_cost[vertex] * self.prob[vertex] / self.prob[tree_idx]
        
        stack = [tree_idx]
        while len(stack):
            cur = stack.pop()
            if cur in slc:
                continue
            cur_tree = self.tag_lookup[cur]
            cost += cur_tree.cost
            if isinstance(cur_tree, Branch):
                stack.append(cur_tree.left.tag)
                stack.append(cur_tree.right.tag)

        assert cost > 0, (tree_idx, slc, self.tag_lookup[tree_idx])
        return cost

        # cost = self.original_total_cost[tree_idx]
        # for vertex in slc:
        #     cost -= self.total_cost[vertex] * (1 - self.prob[vertex] / self.prob[tree_idx])
        # assert cost > 0, (tree_idx, slc)
        # return cost

    def update_costs(self):
        self.push()
        new_cost = {}
        for key in self.total_cost:
            new_cost[key] = self.sliced_cost(key, self.optimal_slice[key])
        self.total_cost = new_cost
        self.optimal_slice = {}




def complete_tree(depth: int):
    if depth > 0:
        return Branch(complete_tree(depth - 1), complete_tree(depth - 1), factor=0.5, cost=1)
    return Leaf(1)

def compute_N_slices(calc: SliceCalculator, N: int) -> Tuple[List[Set[int]], float]:
    calc.enumerate_children()

    for _ in range(N):
        calc.compute_slice(calc.tree.tag)
        calc.update_costs()

    overall_cost = calc.total_cost[calc.tree.tag]

    slices = [{calc.tree.tag}]
    for _ in range(N):
        calc.pop()
        new_slice = set()
        for vertex in slices[-1]:
            new_slice.update(calc.optimal_slice[vertex])
        slices.append(new_slice)

    return slices[1:], overall_cost

# calc = SliceCalculator(T)
# calc.enumerate_children()
# print(calc.compute_slice(T.tag))
# calc.update_costs()
# print(calc.compute_slice(T.tag))

def expected_cost(t: Tree) -> float:
    if isinstance(t, Leaf):
        return t.cost
    return t.p_left * expected_cost(t.left) + t.p_right * expected_cost(t.right) + t.cost


def entropy(T: Tree, slc: Set[int]) -> float:
    calc = SliceCalculator(T)
    calc.enumerate_children()

    total = 0.
    for vertex in slc:
        total -= calc.prob[vertex] * log2(calc.prob[vertex])

    return total


# for i in range(4):
#     print(compute_N_slices(SliceCalculator(A), i))
if __name__ == '__main__':
    T = complete_tree(6)
    
    leaves = [Leaf(2) for _ in range(6)]
    D = Branch(leaves[0], leaves[1], factor=0.25, cost=1)
    E = Branch(leaves[2], leaves[3], factor=0.25, cost=5)
    C = Branch(leaves[4], leaves[5], factor=0.25, cost=10)
    B = Branch(D, E, factor=0.25, cost=10)
    A = Branch(B, C, factor=0.25, cost=1)

    for i in range(3):
        print(f'{i + 1} slices')
        slices, cost = compute_N_slices(SliceCalculator(A), i + 1)
        print(f'Entropy slice: {slices[-1]}')
        print(f'Entropy = {entropy(A, slices[-1])} bits')

# print(expected_cost(A))
