import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from argparse import ArgumentParser
from transformers import *
import logging

from QuestionAnswerer import QuestionAnswerer, Model_dict

def parse_args():
    parser = ArgumentParser(description = 'Ask me a question')
    parser.add_argument(
        '-m',
        '--models',
        '--model-list',
        '--model',
        type = str.lower,
        choices = Model_dict.keys(),
        default = ['falcon-7b'],
        nargs = '+',
        help = 'Which model or models to use',
    )
    parser.add_argument('--device', choices = ['cpu', 'cuda'], default = 'cpu')

    parser.add_argument('questions', nargs = '*', default = 'Where was Obama born?')
    return parser.parse_args()

def main():
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    args = parse_args()
    answerer = QuestionAnswerer(args.models, device = args.device)

    for q in args.questions:
        answers = answerer.query(q)
        for llm, answer in answers.items():
            print(f'\033[1m{llm}\033[0m: {answer}')

    logging.info('Done!')

if __name__ == '__main__':
    main()
