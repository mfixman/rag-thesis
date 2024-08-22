import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from argparse import ArgumentParser, BooleanOptionalAction, RawDescriptionHelpFormatter
import pandas
import torch
import sys

from Models import *
from QuestionAnswerer import *
from Utils import *
from Trainer import *

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

    wandb.init(project = 'question-combinator')

    args = parse_args()
    args.questions_file = args.questions_file[:args.lim_questions]

    context_prompt = 'Answer the following question using the previous context in a few words and with no formatting.'
    data_text = [
        Object \
            .orNothing(thing = row.Thing, category = row.Category, question = row.Base_Question + row.Prefix) \
            .format(prompt = context_prompt, context = row['counterfactual-' + args.model])
        for _, row in args.questions_file.iterrows()
    ]
    target_text = args.question_file['parameric-' + args.model]

    model = Model(args.model, device = args.device)
    wandb.watch(model)
    qa = QuestionAnswerer(model, device = args.device, do_train = True)
    [(data, data_attn), (target, target_attn)] = qa.tokenize_many(data_text, target_text)

    trainer = Trainer(model, data, data_attn, target, target_attn)
    loss = trainer.train()
    logging.info('Finished! Final loss is {loss}')

if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        main()
