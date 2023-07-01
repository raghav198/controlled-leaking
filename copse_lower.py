from holla import ChallahBranch, ChallahTree
from slice_program import challah_to_basic
from slicing_dp import Tree, Branch


class tag_branches:
    def __init__(self):
        self.cur_tag = 0
        
    def __call__(self, tree: Tree):
        if isinstance(tree, Branch):
            self(tree.left)
            self(tree.right)
            tree.tag = self.cur_tag
            self.cur_tag += 1

def generate_copse_data(tree: ChallahTree):
    basic_tree = challah_to_basic(tree)
    tag_branches()(basic_tree)
    
    masks = [level_mask(tree, level) for level in range(depth(tree))]
    matrices = level_matrices(basic_tree, masks)
    return masks[::-1], matrices[::-1] # we compute these root to leaf, but COPSE expects them leaf to root


def generate_copse_cpp(masks: list[list[int]], matrices: list[list[list[int]]], name: str, eq_mask: str, lt_mask: str):
    mask_vectors = ',\n        '.join([str(mask).replace('[', '{').replace(']', '}') for mask in masks])
    
    
    def matrix_to_text(matrix: list[list[int]]):
        rows = ',\n            '.join([str(row).replace('[', '{').replace(']', '}') for row in matrix])
        return f"""
        {{
            {rows}
        }}"""
        
    matrix_code = ',        '.join([matrix_to_text(matrix) for matrix in matrices])
        
    cpp = f"""
#include <copse/model-owner.hpp>

#include "../kernel.hpp"

auto {name}_model() {{
    PtxtModelDescription model;
    model.name = "{name.upper()}";
    model.bits = 8;
    model.k = -1;
    model.thresholds = {{}};
    
    model.level_mask = {{
        {mask_vectors}
    }};
    
    model.level_b2s = {{
       {matrix_code} 
    }};
    
    return model;
}}

std::string COILMaurice::eq_mask() {{ return "{eq_mask}"; }} 
std::string COILMaurice::lt_mask() {{ return "{lt_mask}"; }}
PtxtModelDescription COILMaurice::plaintext_model() {{
    return {name}_model();
}}
CtxtModelDescription* COILMaurice::GenerateModel() {{
    return ModelOwner::GenerateModel();
}}
"""
    return cpp



def depth(tree: ChallahTree | Tree):
    if isinstance(tree, ChallahBranch):
        return max(depth(tree.true), depth(tree.false)) + 1
    if isinstance(tree, Branch):
        return max(depth(tree.left), depth(tree.right)) + 1
    return 0

def num_leaves(tree: ChallahTree | Tree):
    if isinstance(tree, ChallahBranch):
        return num_leaves(tree.true) + num_leaves(tree.false)
    if isinstance(tree, Branch):
        return num_leaves(tree.left) + num_leaves(tree.right)
    return 1 # a single leaf or an array of leaves both count as a single leaf ;)


def level_mask(tree: ChallahTree, depth: int, prev: int = -1) -> list[int]:
    if isinstance(tree, ChallahBranch):
        if depth > 0:
            return level_mask(tree.true, depth - 1, prev=0) + level_mask(tree.false, depth - 1, prev=1)
        return [0] * num_leaves(tree.true) + [1] * num_leaves(tree.false)
    return [prev]
    

def level_matrices(tree: Tree, level_masks: list[list[int]]):
    if not isinstance(tree, Branch):
        return [[[]]]
    width = num_leaves(tree)
    level_parents: list[list[Branch]] = [[tree for _ in range(width)]]
    
    def parents_to_matrix(parents: list[Branch]):
        leaves = num_leaves(tree)
        branches = tree.tag + 1
        
        # in the level matrix, the column corresponds the branch, and the row corresponds to the leaf
        # i.e., many leaves for one branch appear as a column of many 1's in the matrix
        matrix = [[0 for _ in range(branches)] for _ in range(leaves)] # index matrix[row][column]
        for leaf, parent in enumerate(parents):
            matrix[leaf][parent.tag] = 1
            
        return matrix
    
    for d in range(depth(tree) - 1):
        new_level = level_parents[-1][:]
        for leaf in range(width):
            if level_masks[d][leaf] == 1 and isinstance(parent := level_parents[-1][leaf].right, Branch): # this leaf is on the false path
                new_level[leaf] = parent
            elif level_masks[d][leaf] == 0 and isinstance(parent := level_parents[-1][leaf].left, Branch): # this leaf is on the true path
                new_level[leaf] = parent
        level_parents.append(new_level)

    
    return [parents_to_matrix(parents) for parents in level_parents]
