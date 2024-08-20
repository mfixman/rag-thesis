import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import itertools
import logging
import torch
import typing

from torch import LongTensor, FloatTensor

from Models import Model
from typing import Optional, Union

class QuestionAnswerer:
    device: str
    max_length: Optional[int]

    llm: Model

    def __init__(
        self,
        model: Union[str, Model],
        device: str = 'cpu',
        max_length: Optional[int] = None,
    ):
        self.device = device
        self.max_length = max_length

        if type(model) == str:
            model = Model(model, device = device)

        model = typing.cast(Model, model)
        self.llm = model.to(device)

    def query_dict(self, question_dict: dict[tuple[str, str], str]) -> dict[tuple[str, str], tuple[str, float]]:
        answer_list = self.query_logits(list(question_dict.values()))
        return dict(zip(question_dict.keys(), answer_list))

    @torch.no_grad()
    def query_outputs(self, questions: list[str]) -> tuple[LongTensor, FloatTensor]:
        tokens = self.llm.tokenizer(
            questions,
            return_tensors = 'pt',
            return_attention_mask = True,
            padding = True,
        ).to(self.device)

        outputs = self.llm.model.generate(
            input_ids = tokens['input_ids'],
            attention_mask = tokens['attention_mask'],
            max_new_tokens = self.max_length,
            stop_strings = ['.', '\n'],
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

        return outputs.sequences, outputs.logits

    @torch.no_grad()
    def query_logits(self, questions: list[str]) -> list[tuple[str, float]]:
        sequences, logits = self.query_outputs(questions)

        generated = sequences[:, tokens['input_ids'].shape[1]:]
        bad_tokens = torch.tensor(self.llm.tokenizer.convert_tokens_to_ids(['.'] + list(self.llm.tokenizer.special_tokens_map.values()))).to(self.device)
        invalid_mask = torch.isin(generated, bad_tokens)

        answers = [
            self.llm.tokenizer.decode(
                x,
                skip_special_tokens = True,
            ).strip('.\n ' + self.llm.tokenizer.pad_token)
            for x in generated
        ]

        logits = torch.stack(logits, dim = 1).softmax(dim = 2)
        logits[invalid_mask] = 1
        logit_prod = logits.max(dim = 2)[0].prod(dim = 1).cpu().numpy()

        return list(zip(answers, logit_prod))

    @torch.no_grad()
    def query(self, questions: list[str]):
        sequences, _ = self.query_outputs(questions)
        answers = self.llm.tokenizer.batch_decode(
            sequences,
            skip_special_tokens = True,
            clean_up_tokenization_spaces = True,
        )
        return [a.removeprefix(q).strip().strip('.') for q, a in zip(questions, answers)]
