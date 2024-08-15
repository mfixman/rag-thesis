import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from argparse import ArgumentParser, BooleanOptionalAction, RawDescriptionHelpFormatter
from pathlib import Path
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
    default_prompt = 'Answer the following question using as few words as possible. Context: [{context}]; Question: [{question}]; Answer:'

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

    parser.add_argument('--raw-answers', action = 'store_true', help = 'Write the answers from these files')
    parser.add_argument('--empty-context', '--empty', action = 'store_true', default = True, help = 'Whether to use an empty context as base')
    parser.add_argument('--linear-context', '--linear', action = 'store_true', help = 'List of files with equal amount of lines as question file.')
    parser.add_argument('--combined-context', '--combined', action = 'store_true', help = 'List of files whose lines will get intermixed.')
    parser.add_argument('--full-context', '--full', action = 'store_true', help = 'Files with data to inject to RAG extractor.')

    parser.add_argument('--input-files', nargs = '+', default = [], help = 'Alternative way to add files from all --rag-X-files contexts.')
    parser.add_argument('--full-context-shuffles', metavar = 'N', type = int, help = 'If --full-context is specified, shuffle the input N times and return a summary of the results')

    parser.add_argument('--logits', action = BooleanOptionalAction, default = False, help = 'Whether to also output logits')

    parser.add_argument('question_file', type = open, nargs = '?', help = 'File with questions')

    args = parser.parse_args()
    if args.list_models:
        for k, v in [('     \033[1mModel Name', 'Huggingface Model\033[0m')] + list(Model_dict.items()):
            print(f'{k:>15s} | {v:<60s}')

        sys.exit(0)

    if args.question_file is None:
        sys.exit('question_file must be specified!')

    if args.full_context is None and args.full_context_shuffles is not None:
        sys.exit('--full-context-shuffles requires --full-context')

    if args.full_context_shuffles is not None:
        raise NotImplemented('--full-context-shuffles not implemented yet')

    if len(args.models) > 1:
        raise NotImplemented('Multiple models teporarily disabled.')
    args.model = args.models[0]

    args.context = {Path(file).stem: [x.strip() for x in open(file)] for file in args.input_files}
    args.questions = [q.strip() for q in args.question_file if not q.isspace()]
    del args.question_file

    for k, v in args.context.items():
        if len(args.questions) != len(v):
            sys.exit(f'Answer file {k} has a different amount of answers than expected')

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
    if args.full_context:
        rags.append(FullRAG(args.context))
    if args.linear_context:
        for k, v in args.context.items():
            rags.append(LinearRAG({k: v}, args.questions))
    if args.combined_context:
        for c in itertools.permutations(args.context.items()):
            rags.append(LinearRAG(dict(c), args.questions))

    prevs = []
    if args.raw_answers:
        for k, v in args.context.items():
            prevs.append(RawAnswerRAG(k, v, args.questions))

    asker = QuestionAsker(model_names = args.models, prevs = prevs, rags = rags, include_logits = args.logits)
    questions = asker.prepareQuestions(prompt = args.custom_prompt, questions = args.questions)

    answerer = QuestionAnswerer(Model.fromName(args.model), device = args.device, max_length = args.max_length)
    answers = answerer.query_dict(questions)
    print(answers)

    logging.info('Done!')

if __name__ == '__main__':
    main()
