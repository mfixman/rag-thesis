import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from argparse import ArgumentParser, BooleanOptionalAction, RawDescriptionHelpFormatter
import csv
import logging
import random
import ipdb
import itertools

import wandb

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

def main():
    random.seed(0)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    args = parse_args()

    # wandb.init(
    #     entity = 'mfixman-rag-thesis',
    #     project = 'question-combinator',
    #     settings = wandb.Settings(system_sample_seconds = 5),
    # )

    logging.info('Getting questions')
    questions, cat_positions = combine_questions(args.base_questions, args.things, args.lim_questions)
    flips = find_flips(cat_positions, len(questions))

    logging.info(f'About to answer {len(questions) * len(args.models) * (1 + args.counterfactuals)} questions in total.')
    parametric = {}
    counterfactuals = {}
    for model in args.models:
        prompt = 'Answer the following question in a few words and with no formatting.'
        qa = QuestionAnswerer(model, device = args.device, max_length = 15)
        parametric[model] = qa.query([q.format(prompt = prompt) for q in questions])

        if args.counterfactuals:
            cf = [parametric[model][x] for x in flips]

            assert len(questions) == len(cf)
            queries = [
                q.format(prompt = prompt, context = context)
                for q, context in zip(questions, cf)
            ]

            counterfactuals[f'counterfactual-{model}'] = cf
            counterfactuals[f'nonparametric-{model}'] = qa.query(queries)

    logging.info('Writing CSV')
    printParametricCSV(questions, parametric, counterfactuals)

if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        main()
