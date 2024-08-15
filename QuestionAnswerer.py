import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import itertools
import logging
import torch
import sys

from Models import Model
from typing import Optional, Union

class QuestionAnswerer:
    device: str
    max_length: Optional[int]
    short: bool
    return_rest: bool

    llm: Model

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

    def query_dict(self, question_dict: dict[tuple[str, str], str]) -> dict[tuple[str, str], tuple[str, float]]:
        answer_list = self.query(list(question_dict.values()))
        print(type(answer_list))
        print(answer_list[0])
        raise ValueError('asdf')
        return dict(zip(question_dict.keys(), answer_list))

    @torch.no_grad()
    def query(self, questions: list[str]) -> list[tuple[str, float]]:
        tokens = self.llm.tokenizer(
            questions,
            return_tensors = 'pt',
            return_attention_mask = True,
            padding = True,
        ).to(self.device)
        print(tokens['input_ids'][0])
        print(tokens['attention_mask'][0])

        outputs = self.llm.model.generate(
            input_ids = tokens['input_ids'],
            attention_mask = tokens['attention_mask'],
            max_new_tokens = self.max_length,
            stop_strings = '.',
            do_sample = False,
            tokenizer = self.llm.tokenizer,
            output_logits = True,
            return_dict_in_generate = True,
            temperature = None,
            top_p = None,
            pad_token_id = self.llm.tokenizer.pad_token_id,
            eos_token_id = self.llm.tokenizer.eos_token_id,
            bos_token_id = self.llm.tokenizer.bos_token_id,
        )

        answers = self.llm.tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens = True
        )
        print(answers[0])
        sys.exit(0)

        logit_prod = torch.stack(outputs.logits, dim = 1).softmax(dim = 2).max(dim = 2)[0].prod(dim = 1).cpu().numpy()
        return list(zip(answers, logit_prod))
