import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from argparse import ArgumentParser, BooleanOptionalAction
from transformers import *
import logging

from QuestionAnswerer import QuestionAnswerer, Model_dict
from RagEnhancer import *

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
    parser.add_argument('-l', '--max-length', type = int, default = 100, help = 'Max length of answer')

    parser.add_argument('--rag', action = BooleanOptionalAction, default = True, help = 'Whether to enhance the answer with RAG')

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

    rag = EmptyRAG()
    if args.rag:
        rag = RAG()

    for q in args.questions:
        context = rag.retrieve_context(q)

        enhanced_question = f'Context: [{context}]; Question: [{q}]: Answer: '
        answers = answerer.query(enhanced_question, max_length = args.max_length)
        for llm, answer in answers.items():
            print(f'\033[1m{llm}\033[0m: {answer}')

    logging.info('Done!')

if __name__ == '__main__':
    main()
