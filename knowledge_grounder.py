import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from argparse import ArgumentParser
import csv
import logging
import random
import ipdb
import os
import sys
import wandb

from Models import Model_dict
from QuestionAnswerer import QuestionAnswerer
from Utils import print_parametric_csv, LogTimeFilter, combine_questions

def parse_args():
    parser = ArgumentParser(
        description = 'Combines questions and data and optionally provides parametric data'
    )

    parser.add_argument('--debug', action = 'store_true', help = 'Go to IPDB console on exception.')
    parser.add_argument('--lim-questions', type = int, help = 'Question limit')
    parser.add_argument('--device', choices = ['cpu', 'cuda'], default = 'cuda', help = 'Inference device')
    parser.add_argument('--models', type = str.lower, default = [], choices = Model_dict.keys(), nargs = '+', metavar = 'model', help = 'Which model or models to use for getting parametric data')
    parser.add_argument('--offline', action = 'store_true', help = 'Tell HF to run everything offline.')
    parser.add_argument('--rand', action = 'store_true', help = 'Seed randomly')
    parser.add_argument('--max-batch-size', type = int, default = 120, help = 'Maximimum size of batches. All batches contain exactly the same question.')

    parser.add_argument('--per-model', action = 'store_true', help = 'Write one CSV per model in stdout.')
    parser.add_argument('--output-dir', help = 'Return one CSV per model, and save them to this directory.')

    parser.add_argument('--runs-per-question', type = int, default = 1, help = 'How many runs (with random counterfactuals) to do for each question.')

    parser.add_argument('base_questions_file', type = open, help = 'File with questions')
    parser.add_argument('things_file', type = open, help = 'File with things to combine')

    args = parser.parse_args()

    args.base_questions = [x.strip() for x in args.base_questions_file if any(not y.isspace() for y in x)]
    args.things = [{k: v for k, v in p.items()} for p in csv.DictReader(args.things_file)]

    del args.base_questions_file
    del args.things_file

    if args.per_model and args.output_dir:
        raise ValueError('Only one of --per-model and --output-dir can be specified.')

    return args

def main(args):
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger().addFilter(LogTimeFilter())

    if args.offline:
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    else:
        wandb.init(project = 'knowledge-grounder', config = args)

    logging.info('Getting questions')
    questions = combine_questions(args.base_questions, args.things, args.lim_questions)

    if args.output_dir:
        try:
            os.mkdir(args.output_dir)
        except FileExistsError:
            pass

    logging.info(f'About to answer {len(questions) * len(args.models) * 2} questions in total.')
    answers = {}
    for model in args.models:
        if not args.rand:
            random.seed(0)

        qa = QuestionAnswerer(
            model,
            device = args.device,
            max_length = 20,
            max_batch_size = args.max_batch_size,
            runs_per_question = args.runs_per_question,
        )
        model_answers = qa.answerQueries(questions)
        del qa

        if args.output_dir:
            empty = lambda s: sum([x == '' for x in model_answers[s]])
            count = lambda s: sum([x == s for x in model_answers['comparison']])
            logging.info(f"{model}:\t{empty('parametric')} empty parametrics, {empty('counterfactual')} empty counterfactuals, {empty('contextual')} empty contextuals")
            logging.info(f"\t{count('Parametric')} parametrics, {count('Contextual')} contextual, {count('Other')} others")

            model_filename = os.path.join(args.output_dir, model + '.csv')
            with open(model_filename, 'w') as out:
                print_parametric_csv(out, questions, model_answers)

        elif args.per_model:
            print_parametric_csv(sys.stdout, questions, model_answers)
        else:
            answers |= model_answers

    if answers:
        logging.info('Writing CSV')
        print_parametric_csv(sys.stdout, questions, answers)

if __name__ == '__main__':
    args = parse_args()
    if not args.debug:
        main(args)
    else:
        with ipdb.launch_ipdb_on_exception():
            main(args)
