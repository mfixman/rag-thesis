import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from argparse import ArgumentParser, BooleanOptionalAction
from transformers import *
import logging
import sys

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

    parser.add_argument('--empty', type = BooleanOptionalAction, default = True, help = 'Whether to use an empty context as base')
    parser.add_argument('--rag', action = BooleanOptionalAction, default = False, help = 'Whether to enhance the answer with RAG')
    parser.add_argument('--rag-const', help = 'Mock this context for RAG rather than using a RAG extractor.')
    parser.add_argument('--rag-const-file', type = open, help = 'File with data to inject to RAG extractor.')

    parser.add_argument('--dummy', action = BooleanOptionalAction, default = False, help = 'Use dummy dataset for RAG')

    parser.add_argument('question_file', type = open, help = 'File with questions')

    args = parser.parse_args()

    return args

def main():
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    args = parse_args()
    answerer = QuestionAnswerer(args.models, device = args.device)

    rags = []
    if args.empty:
        rags.append(EmptyRAG)
    if args.rag:
        rags.append(RAG(dummy = args.dummy))
    if args.rag_const is not None:
        rags.append(ConstRAG(args.rag_const))
    if args.rag_const_file is not None:
        rags.append(FileRAG(args.rag_const_file))

    for q in args.question_file:
        q = q.strip('\n')
        if q.isspace():
            continue

        print(q)
        for rag in rags:
            context = rag.retrieve_context(q)
            enhanced_question = f'Context: [{context}]; Question: [{q}]. Answer briefly using the previous context and without prompting. Answer:'
            answers = answerer.query(enhanced_question, max_length = args.max_length)

            print(rag.name())
            for llm, answer in answers.items():
                print(f'\033[1m{llm}\033[0m: {answer.removeprefix(enhanced_question)}')

    logging.info('Done!')

if __name__ == '__main__':
    main()
