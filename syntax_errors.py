from argparse import ArgumentParser
import lark

def report_syntax_errors(filename: str):
    p = lark.Lark(open('grammar.lark').read())
    try:
        p.parse(open(filename).read())
    except Exception as e:
        print(f'{e}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()

    report_syntax_errors(args.file)
    
