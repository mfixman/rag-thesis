import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from argparse import ArgumentParser, BooleanOptionalAction, RawDescriptionHelpFormatter
from dataclasses import dataclass
import csv
import logging

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
    question: None | str = None

    def valid(self, question: str) -> bool:
        return f'{{{self.category}}}' in question

    def format(self) -> None | str:
        if self.question is None:
            raise ValueError('Question not set')

        if not self.valid(self.question):
            raise ValueError(f'Invalid question {question} for category {self.category}')
        
        return self.question.format_map({self.category: self.object})

def main():
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    args = parse_args()

    questions = []
    for bq in args.base_questions:
        for obj in args.objects:
            if obj.valid(bq):
                obj.question = bq
                questions.append(obj)

    questions = questions[:args.lim_questions]

    parametric = {}
    for model in args.models:
        qa = QuestionAnswerer(Model(model), device = args.device, max_length = 50)
        parametric[model] = qa.query([q.format() for q in questions])

    writer = csv.DictWriter(sys.stdout, fieldnames = ['Question', 'Category'] + [f'Parametric-{x}' for x in args.models])
    writer.writeheader()
    for question, answers in zip(questions, *parametric.values()):
        param = dict(zip(paramteric.keys(), answers))
        writer.writerrow({'Question': question.format(), 'Category': question.category} | param)

if __name__ == '__main__':
    main()
