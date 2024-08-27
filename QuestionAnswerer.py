import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import itertools
import logging
import torch
import typing

from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

from Models import Model
from typing import Optional, Union, Any
from Utils import Object

import ipdb
import sys


FloatTensor = torch.Tensor
LongTensor = torch.Tensor
BoolTensor = torch.Tensor

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
        self.llm = model

    def query_dict(self, question_dict: dict[tuple[str, str], str]) -> dict[tuple[str, str], tuple[str, float]]:
        answers, logits = self.query(list(question_dict.values()))
        return dict(zip(question_dict.keys(), zip(answers, logits)))

    # [n] -> (n, w, v)
    @torch.no_grad()
    def query(self, questions: list[str]) -> FloatTensor:
        tokens = self.llm.tokenizer(
            questions,
            return_tensors = 'pt',
            return_attention_mask = True,
            padding = True,
        ).to(self.device)

        batch_size = 5000
        chunks = 1 + (tokens['input_ids'].shape[0] * tokens['input_ids'].shape[1]) // batch_size
        input_ids = tokens['input_ids'].chunk(chunks, dim = 0)
        attention_masks = tokens['attention_mask'].chunk(chunks, dim = 0)

        logging.info(f"Answering questions for {len(questions)} × {tokens['input_ids'].shape[1]} = {len(questions) * tokens['input_ids'].shape[1]} in {chunks} chunks.")
        all_logits: list[FloatTensor] = []
        for ids, masks in zip(input_ids, attention_masks):
            outputs = self.llm.model.generate(
                input_ids = ids,
                attention_mask = masks,
                max_new_tokens = self.max_length,
                # stop_strings = ['.', '\n'],
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
            all_logits.append(torch.stack(outputs.logits, dim = 1).softmax(dim = 2))

        return torch.cat(all_logits, dim = 0)

    # (n, w, v) -> (n, w); (n, w)
    def winner(self, logits: FloatTensor) -> tuple[LongTensor, FloatTensor]:
        probs, path = logits.max(dim = 2)
        return path, probs

    # (n, w); (n, w) -> [n]; [n]
    def decode(self, path: LongTensor, probs: FloatTensor) -> tuple[list[str], list[float]]:
        stop_token = self.llm.tokenizer.convert_tokens_to_ids('.')

        ignores = torch.cumsum(path == stop_token, dim = 1) > 0
        path[ignores] = self.llm.tokenizer.pad_token_id
        probs[ignores] = torch.nan

        phrase = self.llm.tokenizer.batch_decode(path, skip_special_tokens = True, clean_up_tokenization_spaces = True)
        avg_probs = list(probs.nanmean(dim = 1).cpu().numpy())

        return phrase, avg_probs

    def gather(self, logits: FloatTensor, ids: LongTensor) -> list[float]:
        assert logits.shape[0] == ids.shape[0]
        assert logits.shape[1] == ids.shape[1]
        assert torch.all((logits >= 0) & (logits < 1))
        assert torch.isclose(logits.sum(dim = 2), torch.ones(logits.shape[0:2]).to(self.device)).all()

        stop_string_ids = self.llm.tokenizer.convert_tokens_to_ids(['.'])

        traces = logits.gather(index = ids.unsqueeze(2), dim = 2).squeeze(2)
        traces[ids == self.llm.tokenizer.pad_token_id] = torch.nan

        for i, trace in zip(ids, traces):
            logging.info(zip(i, trace))

        return traces.nanmean(dim = 1).cpu().numpy()

    @staticmethod
    def streq(a: str, b: str) -> bool:
        a = a.lower().replace('the', '').replace(',', '').strip()
        b = b.lower().replace('the', '').replace(',', '').strip()
        return a[:len(b)] == b[:len(a)]

    def answerCounterfactuals(self, questions: list[Object], counterfactuals = list[Object]) -> dict[str, Any]:
        prompt = 'Answer the following question using the previous context in a few words and with no formatting.'

        output: dict[str, Any] = {}
        output['counterfactual'] = counterfactuals

        queries = [
            q.format(prompt = prompt, context = context)
            for q, context in zip(questions, counterfactuals)
        ]

        logits = self.query(queries)
        cparam, cprobs = self.decode(*self.winner(logits))
        output['ctx_answer'] = cparam
        output['ctx_logits'] = cprobs

        return output

    def answerQueries(self, questions: list[Object], counterfactual_flips = Optional[list[int]]) -> dict[str, Any]:
        prompt = 'Answer the following question in a few words and with no formatting.'

        output: dict[str, Any] = {}

        logits = self.query([q.format(prompt = prompt) for q in questions])
        param, probs = self.decode(*self.winner(logits))
        output['parametric'] = param
        output['param_logits'] = probs

        if counterfactual_flips is not None:
            counterfactuals = [param[x] for x in counterfactual_flips]
            assert len(questions) == len(counterfactuals)
            output |= self.answerCounterfactuals(questions, counterfactuals)

            output['comparison'] = [
                'Parametric' if self.streq(a, p) else
                'Counterfactual' if self.streq(a, c) else
                'Other'
                for p, c, a in zip(output['parametric'], output['counterfactual'], output['ctx_answer'])
            ]

        return output
