import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from argparse import ArgumentParser, BooleanOptionalAction, RawDescriptionHelpFormatter
import pandas
import torch
import sys

from Models import *
from QuestionAnswerer import *
from Utils import *

import wandb

def parse_args():
    parser = ArgumentParser(
        description = 'Fine-tunes a model from the response of question_combinator.py.'
    )

    parser.add_argument('--device', choices = ['cpu', 'cuda'], default = 'cpu', help = 'Device')
    parser.add_argument('--model', type = str.lower, required = True, choices = Model_dict.keys(), metavar = 'model', help = 'Which model to fine-tune')

    parser.add_argument('--lim-questions', type = int, help = 'Question limit')

    parser.add_argument('questions_file', type = pandas.read_csv, help = 'File with questions')

    return parser.parse_args()


def main():
    random.seed(0)
    torch.manual_seed(0)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    args = parse_args()
    args.questions_file = args.questions_file[:args.lim_questions]

    context_prompt = 'Answer the following question using the previous context in a few words and with no formatting.'
    data = [
        Object \
            .orNothing(thing = row.Thing, category = row.Category, question = row.Base_Question) \
            .format(prompt = context_prompt, context = row['counterfactual-' + args.model])
        for _, row in args.questions_file.iterrows()
    ]
    print(data)
    sys.exit(1)

    qa = QuestionAnswerer(model = args.model, device = args.device, do_train = True)

if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        main()
