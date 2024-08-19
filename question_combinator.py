import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from argparse import ArgumentParser, BooleanOptionalAction, RawDescriptionHelpFormatter
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
    parser.add_argument('objects_file', type = open, help = 'File with objects to combine')

    args = parser.parse_args()

    args.base_questions = [x.strip() for x in args.base_questions_file if any(not y.isspace() for y in x)]
    args.objects = [Object(**x) for x in csv.DictReader(args.objects_file)]

    del args.base_questions_file
    del args.objects_file

    return args

@dataclass
class Object:
    object: str
    category: str
    question: Optional[str] = None

    def valid(self, question: str) -> bool:
        return f'{{{self.category}}}' in question

    def format(self, prompt: Optional[str] = None, short: bool = False) -> str:
        if self.question is None:
            raise ValueError('Question not set')

        if not self.valid(self.question):
            raise ValueError(f'Invalid question {question} for category {self.category}')
        
        question = self.question.format_map({self.category: self.object})
        if prompt is not None:
            question = ' '.join([prompt, question])

        if short:
            question = ''.join(question.partition('?')[0:2])

        return question

def main():
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    args = parse_args()

    logging.info('Getting questions')
    questions = []
    for bq in args.base_questions:
        for obj in args.objects:
            if obj.valid(bq):
                questions.append(Object(question = bq, object = obj.object, category = obj.category))

    questions = questions[:args.lim_questions]

    flips = []
    for cat, group in itertools.groupby(enumerate(questions), lambda x: x[1].category):
        nums = [x[0] for x in group]
        flips.extend(
            random.choice(
                list(range(min(nums), x)) + 
                list(range(x, 1 + max(nums)))
            )
            for x in nums
        )

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
