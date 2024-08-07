import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from argparse import ArgumentParser, BooleanOptionalAction, RawDescriptionHelpFormatter
from transformers import *
import logging
import sys

from QuestionAnswerer import QuestionAnswerer, Model_dict
from RagEnhancer import *

def parse_args():
    default_prompt = 'Context: [{context}]; Question: [{question}]. Answer briefly using the previous context and without prompting. Answer:'

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
    parser.add_argument('--rag-const-file', metavar = 'FILE_WITH_CONTEXT', type = open, help = 'File with data to inject to RAG extractor.')

    parser.add_argument('question_file', type = open, nargs = '?', help = 'File with questions')

    args = parser.parse_args()
    if args.list_models:
        for k, v in [('     \033[1mModel Name', 'Huggingface Model\033[0m')] + list(Model_dict.items()):
            print(f'{k:>15s} | {v:<60s}')

        sys.exit(0)

    if args.question_file is None:
        sys.exit('question_file must be specified!')

    return args

def main():
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

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
    if args.rag_const_file is not None:
        rags.append(FileRAG(args.rag_const_file))

    for question in args.question_file:
        question = question.strip('\n')
        if question.isspace():
            continue

        print(question, flush = True)
        for rag in rags:
            context = rag.retrieve_context(question)
            enhanced_question = args.custom_prompt.format(context = context, question = question)
            answers = answerer.query(enhanced_question, max_length = args.max_length, short = True)

            print(rag.name(), flush = True)
            for llm, answer in answers.items():
                print(f'\033[1m{llm}\033[0m: {answer.removeprefix(enhanced_question)}', flush = True)

    logging.info('Done!')

if __name__ == '__main__':
    main()
