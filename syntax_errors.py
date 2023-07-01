from argparse import ArgumentParser
import lark

if __name__ == '__main__':
    p = lark.Lark(open('grammar.lark').read())
    parser = ArgumentParser()

    parser.add_argument('file')
    args = parser.parse_args()

    p.parse(open(args.file).read())
