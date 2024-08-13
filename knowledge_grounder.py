import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from argparse import ArgumentParser, BooleanOptionalAction, RawDescriptionHelpFormatter
import csv
import itertools
import logging
import re
import sys

from Models import *
from QuestionAnswerer import *
from QuestionAsker import *
from RagEnhancer import *

def parse_args():
    default_prompt = 'Context: [{context}]; Question: [{question}]. Answer as briefly as possible. Answer:'

    parser = ArgumentParser(
        description = 'Ask me a question',
        formatter_class = RawDescriptionHelpFormatter,
        epilog = f'''
Use the --list-models option for the full list of supported models.

Default prompt:
```
{default_prompt}
```

Example usage: 

# Test llama and falcon2 on the questions in
# datas/questions.txt using both no context and
# the counterfactuals fount in datas/counterfactuals.txt.
python models.py                               \\
    --device cuda                              \\
    --models llama falcon2                     \\
    --empty-context                            \\
    --rag-const-file datas/counterfactuals.txt \\
    datas/questions.txt
'''
    )
    parser.add_argument(
        '--models',
        type = str.lower,
        choices = Model_dict.keys(),
        default = ['llama'],
        nargs = '+',
        metavar = 'model',
        help = 'Which model or models to use',
    )
    parser.add_argument('--list-models', action = 'store_true', help = 'List all available models')

    parser.add_argument('--device', choices = ['cpu', 'cuda'], default = 'cuda')
    parser.add_argument('-l', '--max-length', type = int, default = 100, help = 'Max length of answer')

    parser.add_argument('--custom-prompt', metavar = 'PROMPT', default = default_prompt, help = 'Use a custom prompt for the questions instead of the default one. {context} and {question} fill to the context and question, respectively')

    parser.add_argument('--rag', action = 'store_true', default = False, help = 'Whether to enhance the answer with RAG')
    parser.add_argument('--rag-dummy', action = 'store_true', default = False, help = 'Use dummy dataset for RAG')

    parser.add_argument('--raw-answers', nargs = '*', help = 'Write the answers from these files')
    parser.add_argument('--empty-context', '--empty', action = 'store_true', default = True, help = 'Whether to use an empty context as base')
    parser.add_argument('--linear-context', '--linear', metavar = 'FILE_WITH_CONTEXT', nargs = '*', help = 'List of files with equal amount of lines as question file.')
    parser.add_argument('--combined-context', '--combined', metavar = 'FILE_WITH_CONTEXT', nargs = '*', help = 'List of files whose lines will get intermixed.')
    parser.add_argument('--full-context', '--full', metavar = 'FILES_WITH_CONTEXT', nargs = '*', help = 'Files with data to inject to RAG extractor.')

    parser.add_argument('--input-files', nargs = '*', help = 'Alternative way to add files from all --rag-X-files contexts.')
    parser.add_argument('--rag-const-shuffles', metavar = 'N', type = int, help = 'If --full-context is specified, shuffle the input N times and return a summary of the results')

    parser.add_argument('--logits', action = BooleanOptionalAction, default = False, help = 'Whether to also output logits')

    parser.add_argument('question_file', type = open, nargs = '?', help = 'File with questions')

    args = parser.parse_args()
    if args.list_models:
        for k, v in [('     \033[1mModel Name', 'Huggingface Model\033[0m')] + list(Model_dict.items()):
            print(f'{k:>15s} | {v:<60s}')

        sys.exit(0)

    if args.question_file is None:
        sys.exit('question_file must be specified!')

    if args.full_context is None and args.rag_const_shuffles is not None:
        sys.exit('--rag-const-shuffles requires --full-context')

    if args.rag_const_shuffles is not None:
        raise NotImplemented('--rag-const-shuffles not implemented yet')

    for p in [args.full_context, args.linear_context, args.raw_answers, args.combined_context]:
        if p == []:
            if args.input_files is None:
                sys.exit('Empty list of files in an argument. Did you forget --input-files?')

            for f in args.input_files:
                p.append(f)

        files = [open(x) for x in p]
        p.clear()
        for f in files:
            p.append([x.strip() for x in list(f)])

    args.questions = [q.strip() for q in args.question_file]
    del args.question_file

    return args

def main():
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.info('Starting')

    args = parse_args()

    rags = []
    if args.empty_context:
        rags.append(EmptyRAG())
    if args.rag:
        rags.append(RAG(dummy = False))
    if args.rag_dummy:
        rags.append(RAG(dummy = True))
    if args.full_context is not None:
        rags.append(FileRAG(args.full_context))
    if args.linear_context is not None:
        rags.append(LinearRAG(args.linear_context, args.questions))
    if args.combined_context is not None:
        for files in itertools.permutations(self.combined_context):
            rags.append(LinearRAG(files, args.questions))

    answerer = QuestionAnswerer(Model.fromNames(args.models), device = args.device, max_length = args.max_length)
    asker = QuestionAsker(models = args.models, rags = rags, answer_files = args.raw_answers, include_logits = args.logits)

    questions = [q.strip() for q in args.questions if not q.isspace()]
    asker.findAnswers(answerer, questions, custom_prompt = args.custom_prompt)

    logging.info('Done!')

if __name__ == '__main__':
    main()
