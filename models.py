import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from argparse import ArgumentParser, BooleanOptionalAction, RawDescriptionHelpFormatter
from pathlib import Path
from transformers import *
import csv
import itertools
import logging
import re
import sys

from QuestionAnswerer import QuestionAnswerer, Model_dict
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

    parser.add_argument('--empty', '--empty-context', action = BooleanOptionalAction, default = True, help = 'Whether to use an empty context as base')
    parser.add_argument('--rag', action = BooleanOptionalAction, default = False, help = 'Whether to enhance the answer with RAG')
    parser.add_argument('--rag-dummy', action = BooleanOptionalAction, default = False, help = 'Use dummy dataset for RAG')
    parser.add_argument('--rag-const', metavar = 'CONTEXT', help = 'Mock this context for RAG rather than using a RAG extractor.')
    parser.add_argument('--rag-const-files', metavar = 'FILE_WITH_CONTEXT', type = open, nargs = '*', help = 'Files with data to inject to RAG extractor.')
    parser.add_argument('--rag-lined-files', metavar = 'FILE_WITH_LINES', type = open, nargs = '*', help = 'List of files with equal amount of lines as question file.')
    parser.add_argument('--output-format', choices = ['text', 'csv'], default = 'csv', help = 'Format of the output')

    parser.add_argument('--write-answers', nargs = '*', type = open, help = 'Write the answers from these files')
    parser.add_argument('--input-files', nargs = '*', help = 'Alternative way to add files from all --rag contexts')

    parser.add_argument('--logits', action = BooleanOptionalAction, default = False, help = 'Whether to also output logits')

    parser.add_argument('question_file', type = open, nargs = '?', help = 'File with questions')

    args = parser.parse_args()
    if args.list_models:
        for k, v in [('     \033[1mModel Name', 'Huggingface Model\033[0m')] + list(Model_dict.items()):
            print(f'{k:>15s} | {v:<60s}')

        sys.exit(0)

    if args.question_file is None:
        sys.exit('question_file must be specified!')

    for p in [args.rag_const_files, args.rag_lined_files, args.write_answers]:
        if p != []:
            continue

        if args.input_files is None:
            sys.exit('Empty list of files in an argument. Did you forget --input-files?')

        for f in args.input_files:
            p.append(open(f))

    args.questions = [q.strip('\n') for q in args.question_file]
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
    answerer = QuestionAnswerer(args.models, device = args.device)

    rags = []
    if args.empty:
        rags.append(EmptyRAG())
    if args.rag:
        rags.append(RAG(dummy = False))
    if args.rag_dummy:
        rags.append(RAG(dummy = True))
    if args.rag_const is not None:
        rags.append(ConstRAG(args.rag_const))
    if args.rag_const_files is not None:
        rags.append(FileRAG(args.rag_const_files))
    if args.rag_lined_files is not None:
        rags.append(LinedRAG(args.rag_lined_files, args.questions))

    writer = None
    if args.output_format == 'csv':
        fieldnames = [f'{rag.name()}_{llm}' for llm in args.models for rag in rags]
        if args.logits:
            fieldnames = list(itertools.chain(*[[f'{x}', f'{x}_logits'] for x in fieldnames]))

        if args.write_answers is not None:
            fieldnames = [Path(x.name).stem for x in args.write_answers] + fieldnames

        writer = csv.DictWriter(
            sys.stdout,
            fieldnames = ['Question'] + fieldnames,
            extrasaction = 'ignore',
            dialect = csv.unix_dialect,
            quoting = csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
    else:
        raise NotImplemented('Text format not implemented yet')

    if args.write_answers is None:
        args.write_answers = []

    filenames = [Path(x.name).stem for x in args.write_answers]
    for question, *answers in zip(args.questions, *args.write_answers):
        question = question.strip()
        if question.isspace():
            continue

        answers = [re.sub(r'.*(was by|was on|is|in) +', '', a.strip()) for a in answers]

        results = {'Question': question} | dict(zip(filenames, answers))
        for rag in rags:
            context = rag.retrieve_context(question)
            enhanced_question = args.custom_prompt.format(context = context, question = question)
            answers, rest = answerer.query(enhanced_question, max_length = args.max_length, short = True, return_rest = True)

            for llm, answer in answers.items():
                name = f'{rag.name()}_{llm}'
                results[name] = answer

                if args.logits:
                    results[f'{name}_logits'] = round(rest[llm]['logit_prod'], 2)

        writer.writerow({'Question': question} | results)

    logging.info('Done!')

if __name__ == '__main__':
    main()
