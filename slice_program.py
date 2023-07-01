from holla import ChallahArray, ChallahBranch, ChallahTree, ChallahLeaf, ChallahVar, compile, pprint
from slicing_dp import Branch, Leaf, Tree, SliceCalculator, compute_N_slices
from greedy_entropy import entropy_slice


def cost(leaf: ChallahLeaf | list[ChallahLeaf], op_costs={'+': 1, '-': 1, '*': 10}) -> float:
    if isinstance(leaf, list):
        return sum(map(cost, leaf))
    if isinstance(leaf, ChallahArray):
        return sum(map(cost, leaf.elems))
    if isinstance(leaf, ChallahVar):
        return 0.001
    return op_costs[leaf.op] + cost(leaf.left, op_costs) + cost(leaf.right, op_costs)


def challah_to_basic(tree: ChallahTree) -> Tree:
    if isinstance(tree, ChallahBranch):
        ret = Branch(challah_to_basic(tree.true), challah_to_basic(tree.false), 0.5, cost(tree.left) + cost(tree.right), backref=tree)
    else:
        ret: Tree = Leaf(cost(tree), backref=tree)
    return ret


def prefix_to(tree: Tree, slc: set[int]) -> Tree:
    if tree.tag in slc:
        return Leaf(0.001, backref=tree.backref)
    if isinstance(tree, Leaf):
        return tree
    return Branch(prefix_to(tree.left, slc), prefix_to(tree.right, slc), tree.factor, tree.cost, tree.tag, tree.backref)


def get_layer(tree: Tree, slc: set[int], calc: SliceCalculator) -> tuple[Tree, list[Tree]]:
    layer = [calc.tag_lookup[tag] for tag in slc]
    # print('original tree: ')
    # basic_pprint(tree)
    # print('slice: ')
    # print(slc)
    # print('prefixed tree: ')
    # basic_pprint(prefix_to(tree, slc))
    return prefix_to(tree, slc), layer


def basic_pprint(tree: Tree, depth: int = 0, start=''):
    if isinstance(tree, Branch):
        print(f'{" " * depth * 4}{start}{tree.tag}{"+" if tree.backref is not None else "-"}')
        basic_pprint(tree.left, depth + 1, start='\u2190 ')
        basic_pprint(tree.right, depth + 1, start='\u2192 ')
    else:
        print(' ' * depth * 4 + start + str(tree.tag) + ("+" if tree.backref is not None else "-"))


def compute_slices(num_slices: int, entropy_limit: float, prog: ChallahTree):

    tree = challah_to_basic(prog)

    num_slices -= 1

    # compute the entropy slice first
    bottom, calc = entropy_slice(tree, entropy_limit)
    # basic_pprint(tree)
    # print(f'Entropy slice: {bottom}')

    new_tree, bottom_layer = get_layer(tree, bottom, calc)
    # print('new tree: ')

    new_calc = SliceCalculator(new_tree)
    new_calc.enumerate_children()  # the backrefs should still be fine
    # basic_pprint(new_tree)
    slices, _ = compute_N_slices(new_calc, num_slices)

    # print('slices: ', slices)
    layers = [bottom_layer]
    for slc in slices:
        new_tree, layer = get_layer(new_tree, slc, new_calc)
        layers.append(layer)
    layers.append([new_tree])
    return layers


def abstract_invoke(prog: ChallahTree, ref: ChallahTree, tag: int):
    if prog == ref:
        return ChallahVar(f'__INVOKE__({tag})')
    if not isinstance(prog, ChallahBranch):
        return prog
    return ChallahBranch(prog.left, prog.lt, prog.right, abstract_invoke(prog.true, ref, tag), abstract_invoke(prog.false, ref, tag))


def get_interactive_layers(prog: ChallahTree, num_slices: int, entropy_limit: float):
    layers = compute_slices(num_slices, entropy_limit, prog)
    prog_layers: list[list[ChallahTree]] = []
    prog_layers_unab: list[list[ChallahTree]] = []
    for layer in layers:
        prog_layer = []
        prog_layer_unab = []
        for tree in layer:
            if tree.backref is None:
                continue

            # first, remove everything from the previous layers
            cur_piece = tree.backref
            if prog_layers:
                for i, p in enumerate(prog_layers_unab[-1]):
                    cur_piece = abstract_invoke(cur_piece, p, i)

            prog_layer.append(cur_piece)
            # keep the unabstracted references around to use when deleting other stuff?
            # TODO: think of a less cursed way to do this. Or don't. I couldn't care less.
            prog_layer_unab.append(tree.backref)

        prog_layers.append(prog_layer)
        prog_layers_unab.append(prog_layer_unab)
    return prog_layers, layers


if __name__ == '__main__':
    # hll = gcd(PitaVarExpr('a'), PitaVarExpr('b'), 7)

    from sys import argv
    from pita_parser import expr

    hll = expr.parse_file(open(argv[1]), parse_all=True)[0]

    print('-' * 10)
    print(hll)
    print('-' * 10)

    prog = compile(hll)
    # pprint(prog)
    num_slices = 2
    entropy_limit = 4
    prog_layers, layers = get_interactive_layers(prog, num_slices, entropy_limit)

    # while True:
    #     try:
    #         prog_layers = get_interactive_layers(prog, num_slices, entropy_limit)
    #         break
    #     except AssertionError as e:
    #         if 'no room' in str(e):
    #             print(f'Cannot fit {num_slices} layers above computed entropy slice, decrementing and retrying...')
    #             num_slices -= 1
    #             continue
    #         else:
    #             raise e

    for layer in prog_layers[::-1]:
        for i, prog in enumerate(layer):
            print(f'---{i}---')
            pprint(prog)
        input()

    print('== BASIC MODE ==')
    for layer in layers[::-1]:
        for i, prog in enumerate(layer):
            print(f'---{i}---')
            basic_pprint(prog)
        input()
