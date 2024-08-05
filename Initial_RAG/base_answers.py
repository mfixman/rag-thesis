import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from argparse import ArgumentParser, BooleanOptionalAction, FileType
import logging
import sys

from QuestionAnswerer import QuestionAnswerer, Model_dict

def parse_args():
    parser = ArgumentParser(description = 'Ask a list of question about a list of people.')
    parser.add_argument(
        '-m',
        '--models',
        '--model-list',
        '--model',
        type = str.lower,
        choices = Model_dict.keys(),
        default = ['llama'],
        nargs = '+',
        help = 'Which model or models to use',
    )
    parser.add_argument('--device', choices = ['cpu', 'cuda'], default = 'cpu')
    parser.add_argument('question_file', type = FileType('r'))
    parser.add_argument('person_file', type = FileType('r'))

    return parser.parse_args()

def main():
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    args = parse_args()

    people = [x.strip() for x in args.person_file]
    questions = [x.strip() for x in args.question_file]

    answerer = QuestionAnswerer(args.models, device = args.device)

    for person in people:
        for question in questions:
            q = question.format(person = person)
            enhanced = f'Answer the following question briefly and do not write anything afterwards. Question: {q}. Answer: '

            answers = answerer.query(enhanced, max_length = 50)
            for llm, answer in answers.items():
                print(f'\033[1m{llm}\033[0m: {answer}')

if __name__ == '__main__':
    main()
