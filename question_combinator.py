import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from argparse import ArgumentParser, BooleanOptionalAction, RawDescriptionHelpFormatter
import csv
import logging
import random
import ipdb
import itertools
import os

import wandb

from Models import *
from QuestionAnswerer import *
from Utils import *
from RagEnhancer import *

def parse_args():
    parser = ArgumentParser(
        description = 'Combines questions and data and optionally provides parametric data'
    )

    parser.add_argument('--no-except', action = 'store_true', help = 'Do not go to IPDB console on exception.')
    parser.add_argument('--lim-questions', type = int, help = 'Question limit')
    parser.add_argument('--device', choices = ['cpu', 'cuda'], default = 'cpu', help = 'Inference device')
    parser.add_argument('--models', type = str.lower, default = [], choices = Model_dict.keys(), nargs = '+', metavar = 'model', help = 'Which model or models to use for getting parametric data')
    parser.add_argument('--counterfactuals', action = 'store_true', help = 'Whether to include counterfactuals in final CSV')
    parser.add_argument('--offline', action = 'store_true', help = 'Tell HF to run everything offline.')
    parser.add_argument('--rand', action = 'store_true', help = 'Seed randomly')

    parser.add_argument('base_questions_file', type = open, help = 'File with questions')
    parser.add_argument('things_file', type = open, help = 'File with things to combine')

    args = parser.parse_args()

    args.base_questions = [x.strip() for x in args.base_questions_file if any(not y.isspace() for y in x)]
    args.things = [{k: v for k, v in p.items()} for p in csv.DictReader(args.things_file)]

    del args.base_questions_file
    del args.things_file

    return args

def main(args):
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if not args.rand:
        random.seed(0)

    if args.offline:
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

    logging.info('Getting questions')
    questions, cat_positions = combine_questions(args.base_questions, args.things, args.lim_questions)
    flips = None
    if args.counterfactuals:
        flips = find_flips(cat_positions, len(questions))

    logging.info(f'About to answer {len(questions) * len(args.models) * (1 + args.counterfactuals)} questions in total.')
    answers = {}
    for model in args.models:
        qa = QuestionAnswerer(model, device = args.device, max_length = 20)
        answers |= {
            f'{k}-{model}': v
            for k, v in qa.answerQueries(questions, flips).items()
        }
        del qa

    logging.info('Writing CSV')
    printParametricCSV(questions, answers)

if __name__ == '__main__':
    args = parse_args()
    if args.no_except:
        main(args)
    else:
        with ipdb.launch_ipdb_on_exception():
            main(args)
