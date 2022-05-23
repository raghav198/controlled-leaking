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
            return {tree_idx}

        left_slice = self.compute_slice(tree.left.tag)
        right_slice = self.compute_slice(tree.right.tag)

        def check_lifts(slc: Set[int], inclusion: float, root: int):
            print(f'The slice: {slc}')
            triples: List[int] = []
            for key in self.children:
                if set(self.children[key]).issubset(slc):
                    triples.append(key)

            while len(triples):
                stuff_added: Set[int] = set()
                for A in triples:
                    A_tree = self.tag_lookup[A]
                    assert isinstance(A_tree, Branch)

                    print(f'\tIs it worth it to merge ({A_tree.left.tag}, {A_tree.right.tag}) -> {A}?')

                    i = inclusion
                    p = self.prob[A] / self.prob[root]
                    x = A_tree.p_left

                    A_cost = A_tree.cost
                    B_cost = self.total_cost[A_tree.left.tag]
                    C_cost = self.total_cost[A_tree.right.tag]

                    print(f'\t\t(i, p, x) = ({i:.2f}, {p:.2f}, {x:.2f})')
                    print(f'\t\t(A, B, C) = {A_cost, B_cost, C_cost}')

                    # is it worth it to pull up?
                    thresh = A_cost / (p * (A_cost + (1 - x) * B_cost + x * C_cost))
                    if i < thresh:
                        print(f'\t\tIts worth it!')
                        slc.remove(A_tree.left.tag)
                        slc.remove(A_tree.right.tag)
                        slc.add(A)
                        stuff_added.add(A)

                triples = []
                for key in self.children:
                    children = set(self.children[key])
                    if children.intersection(stuff_added) and children.issubset(slc):
                        triples.append(key)

            print(f'Done lifting...slice is: {slc}')
            return slc

        print(f'Building the slice for {tree_idx}')
        print(f'Lifting left subslice ({tree.left.tag})')
        left_lifted = check_lifts(left_slice, tree.p_left, tree.left.tag)
        print(f'Lifting right subslice ({tree.right.tag})')
        right_lifted = check_lifts(right_slice, tree.p_right, tree.right.tag)

        return left_lifted | right_lifted

leaves = [Leaf(2) for _ in range(6)]
D = Branch(leaves[0], leaves[1], factor=0.25, cost=1)
E = Branch(leaves[2], leaves[3], factor=0.25, cost=5)
C = Branch(leaves[4], leaves[5], factor=0.25, cost=10)
B = Branch(D, E, factor=0.25, cost=10)
A = Branch(B, C, factor=0.25, cost=1)


calc = SliceCalculator(A)
calc.enumerate_children()
print(calc.compute_slice(A.tag))