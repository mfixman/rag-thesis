from transformers import RagConfig, RagTokenizer, RagRetriever, RagSequenceForGeneration
from pathlib import Path
import itertools
import logging
import torch

from collections.abc import Iterator

class RAG:
    def __init__(self, rag_name: str = 'facebook/rag-sequence-nq', device: str = 'cpu', dummy = False):
        self.rag_name = rag_name

        if dummy:
            logging.warn('Warning: Using dummy dataset')

        logging.info('Getting RAG config')
        self.config = RagConfig.from_pretrained(rag_name)

        logging.info('Getting RAG tokenizers')
        self.tokenizer = RagTokenizer.from_pretrained(rag_name)

        logging.info('Getting RAG retriever')
        self.retriever = self.retry(
            25,
            lambda: RagRetriever.from_pretrained(
                rag_name,
                index_name = 'compressed' if not dummy else 'exact',
                use_dummy_dataset = dummy,
            )
        )

        logging.info('Getting RAG model')
        self.model = RagSequenceForGeneration.from_pretrained(
            rag_name,
            retriever = self.retriever,
        ).to(device)
        self.model.eval()

        self.device = device

    @staticmethod
    def retry(retries: int, func):
        for attempts in range(1, retries + 1):
            try:
                return func()
            except Exception as e:
                logging.warn(f'Retry failed in attempt {attempts}/{retries}): {e}')

        logging.error('Too many errors. Giving up.')
        raise

    @torch.no_grad()
    def retrieve_context(self, question):
        logging.info('Preparing RAG batch')
        input_dict = self.tokenizer.prepare_seq2seq_batch(question, return_tensors = 'pt')

        logging.info('Generating answer')
        generated = self.model.generate(input_ids = input_dict['input_ids'])

        logging.info('Decoding answer')
        answer = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens = True
        )
        return answer[0]

    def yield_contexts(self, question: str) -> Iterator[tuple[str, str]]:
        yield (self.name(), self.retrieve_context(question))

    def name(self) -> str:
        return f'RAG {self.rag_name}'

    def names(self) -> list[str]:
        return [self.name()]

class ConstRAG(RAG):
    def __init__(self, const, **kwargs):
        self.const = const

    def retrieve_context(self, question):
        return self.const

    def name(self):
        return f'Constant data RAG'

class FileRAG(ConstRAG):
    def __init__(self, answer_files, **kwargs):
        self.filenames = [Path(x.name).stem for x in answer_files]
        self.const = '; '.join(x.strip() for x in itertools.chain(*answer_files))

    def name(self):
        return 'all_' + '-'.join(self.filenames)

class LinedRAG(RAG):
    answers: dict[str, list[str]]

    def __init__(self, answer_files: list, question_list: list[str]):
        self.filenames = [Path(x.name).stem for x in answer_files]
        self.answers = {q: [x.strip() for x in v] for q, v in zip(question_list, zip(*answer_files))}

    def retrieve_context(self, question: str) -> str:
        return '; '.join(self.answers[question])

    def name(self) -> str:
        return 'lined_' + '-'.join(self.filenames)

class IntermixedRAG(LinedRAG):
    def yield_contexts(self, question: str) -> Iterator[tuple[str, str]]:
        for name, answers in zip(self.names(), itertools.permutations(self.answers[question])):
            yield (name, '; '.join(answers))

    def names(self) -> list[str]:
        return ['lined_' + '-'.join(fs) for fs in itertools.permutations(self.filenames)]

class EmptyRAG(RAG):
    def __init__(self, *args, **kwargs):
        pass

    def retrieve_context(self, question):
        return ''

    def name(self):
        return 'base'
