import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from argparse import ArgumentParser, BooleanOptionalAction, RawDescriptionHelpFormatter
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import csv
import logging
import random
import ipdb

from Models import *
from QuestionAnswerer import *
from Utils import *
from RagEnhancer import *

def parse_args():
    parser = ArgumentParser(
        description = 'Combines questions and data and optionally provides parametric data'
    )

    parser.add_argument('--lim-questions', type = int, help = 'Question limit')
    parser.add_argument('--device', choices = ['cpu', 'cuda'], default = 'cpu', help = 'Inference device')
    parser.add_argument('--models', type = str.lower, default = [], choices = Model_dict.keys(), nargs = '+', metavar = 'model', help = 'Which model or models to use for getting parametric data')
    parser.add_argument('--counterfactuals', action = 'store_true', help = 'Whether to include counterfactuals in final CSV')

    parser.add_argument('base_questions_file', type = open, help = 'File with questions')
    parser.add_argument('things_file', type = open, help = 'File with things to combine')

    args = parser.parse_args()

    args.base_questions = [x.strip() for x in args.base_questions_file if any(not y.isspace() for y in x)]
    args.things = [{k: v for k, v in p.items()} for p in csv.DictReader(args.things_file)]

    del args.base_questions_file
    del args.things_file

    return args

@dataclass
class Object:
    thing: str
    category: str
    question: str

    @staticmethod
    def orNothing(thing: str, category: str, question: str) -> Optional['Object']:
        if not f'{{{category}}}' in question:
            return None

        return Object(thing, category, question)

    def format(self, prompt: Optional[str] = None, short: bool = False) -> str:
        question = self.question.format_map({self.category: self.thing})
        if prompt is not None:
            question = ' '.join([prompt, question])

        if short:
            question = ''.join(question.partition('?')[0:2])

        return question

def main():
    random.seed(0)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    args = parse_args()

    logging.info('Getting questions')
    questions = []
    cat_positions = defaultdict(lambda: set())
    p = 0
    for bq in args.base_questions:
        if len(questions) == args.lim_questions:
            break

        for thing in args.things:
            p += 1
            obj = Object.orNothing(thing = thing['thing'], category = thing['category'], question = bq)
            if obj is None:
                continue

            questions.append(obj)
            cat_positions[obj.category].add(len(questions) - 1)

            if len(questions) == args.lim_questions:
                break

    flips = [None for _ in questions]
    for cat, values in cat_positions.items():
        for v in values:
            flips[v] = random.choice(list(values - {v}))

    assert all(x is not None for x in flips)

    logging.info(f'Answering {len(questions)} questions')
    parametric = {}
    counterfactual = {}
    for model in args.models:
        prompt = 'Answer the following question in a few words and with no formatting.'
        qa = QuestionAnswerer(model, device = args.device, max_length = 15)
        parametric[model] = qa.query([q.format(prompt = prompt) for q in questions])
        counterfactual[f'counterfactual-{model}'] = [parametric[model][x] for x in flips]

    logging.info('Writing CSV')

    fieldnames = ['Category', 'Question'] + list(parametric.keys())
    if args.counterfactuals:
        fieldnames += list(counterfactual.keys())

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames = fieldnames,
        extrasaction = 'ignore',
        dialect = csv.unix_dialect,
        quoting = csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    for question, *answers in zip(questions, *parametric.values(), *counterfactual.values()):
        param = dict(zip(parametric.keys(), answers))
        counter = dict(zip(counterfactual.keys(), answers[len(parametric):]))
        writer.writerow({'Category': question.category, 'Question': question.format(short = True)} | param | counter)

if __name__ == '__main__':
    main()
