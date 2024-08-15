import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import itertools
import logging
import torch

from Models import Model
from typing import Optional, Union

class QuestionAnswerer:
    device: str
    max_length: Optional[int]
    short: bool
    return_rest: bool

    llms: list[Model]

    def __init__(
        self,
        model: Model,
        device: str = 'cpu',
        max_length: Optional[int] = None,
        short: bool = True,
        return_rest: bool = True
    ):
        self.device = device
        self.max_length = max_length
        self.short = short
        self.return_rest = return_rest

        self.llm = model.to(device)

    def query_dict(self, question_dict: dict[tuple[str, str], str]) -> list[tuple[str, float]]:
        return self.query_new(list(question_dict.values()))

    @torch.no_grad()
    def query_new(self, questions: list[str]) -> list[tuple[str, float]]:
        logging.info(type(questions))
        tokens = self.llm.tokenizer(
            questions,
            return_tensors = 'pt',
            return_attention_mask = True,
            padding = True,
        ).to(self.device)
        outputs = self.llm.model.generate(
            input_ids = tokens['input_ids'],
            attention_mask = tokens['attention_mask'],
            do_sample = False,
            max_new_tokens = self.max_length,
            output_logits = True,
            renormalize_logits = True,
            stop_strings = '.',
            temperature = None,
            top_k = None,
            top_p = None,
            tokenizer = self.llm.tokenizer,
        )
        answers = self.llm.tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens = True
        )

        logit_prod = outputs.logits.softmax(dim = 2).max(dim = 2)[0].prod(dim = 1)
        logging.info(f'{tokens.shape=} {outputs.shape=} {answers.shape=} {logit_prod.shape=}')
        raise ValueError('goodbye!')

        return []
