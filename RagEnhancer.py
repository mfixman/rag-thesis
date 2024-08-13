from transformers import RagConfig, RagTokenizer, RagRetriever, RagSequenceForGeneration
from pathlib import Path
import itertools
import logging
import torch
import re

from collections.abc import Iterator
from typing import IO

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

    def name(self) -> str:
        return f'RAG {self.rag_name}'

class ConstRAG(RAG):
    def __init__(self, const, **kwargs):
        self.const = const

    def retrieve_context(self, question):
        return self.const

    def name(self):
        return f'Constant data RAG'

class FullRAG(ConstRAG):
    def __init__(self, context: dict[str, str]):
        self.filenames = context.keys()
        self.const = '; '.join(itertools.chain(context.values()))

    def name(self):
        return 'all-' + '_'.join(self.filenames)

class LinearRAG(RAG):
    answers: dict[str, list[str]]

    def __init__(self, context: dict[str, str], question_list: list[str]):
        self.filenames = context.keys()
        self.answers = {q: rs for q, *rs in zip(question_list, *context.values())}

    def retrieve_context(self, question: str) -> str:
        return '; '.join(self.answers[question])

    def name(self) -> str:
        return 'linear-' + '_'.join(self.filenames)

class RawAnswerRAG(RAG):
    def __init__(self, filename: str, answers: list[str], question_list: list[str]):
        self.filename = filename
        self.answers = dict(zip(question_list, answers))

    def retrieve_context(self, question: str) -> str:
        answer = self.answers[question]
        return re.sub(r'.*(was by|was on|is|in) +', '', answer)

    def name(self) -> str:
        return self.filename

class EmptyRAG(RAG):
    def __init__(self, *args, **kwargs):
        pass

    def retrieve_context(self, question):
        return ''

    def name(self):
        return 'base'
