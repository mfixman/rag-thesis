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
from Utils import Object, findFlips2, chunkByQuestion

from collections import defaultdict
from transformers import BatchEncoding

import ipdb
import sys

FloatTensor = torch.Tensor
LongTensor = torch.Tensor
BoolTensor = torch.Tensor

class QuestionAnswerer:
    device: str
    max_length: int
    llm: Model

    def __init__(
        self,
        model: Union[str, Model],
        device: str = 'cpu',
        max_length: Optional[int] = None,
    ):
        self.device = device
        self.max_length = max_length or 100

        if type(model) == str:
            model = Model.fromName(model, device = device)

        model = typing.cast(Model, model)
        self.llm = model

        stop_tokens = {'.', '\n'}
        self.stop_token_ids = torch.tensor([
            v
            for k, v in self.llm.tokenizer.get_vocab().items()
            if
                k in ['<start_of_turn>', '<end_of_turn>', self.llm.tokenizer.special_tokens_map['eos_token']] or
                not stop_tokens.isdisjoint(self.llm.tokenizer.decode(v))
        ]).to(self.device)

    # [n] -> (n, w)
    def tokenise(self, phrases: list[str]) -> BatchEncoding:
        return self.llm.tokenizer(
            phrases,
            return_tensors = 'pt',
            return_attention_mask = True,
            padding = True,
        ).to(self.device)

    # (n, w) -> (n, w)
    def batch_encode(self, tokens: LongTensor) -> BatchEncoding:
        attention_mask = tokens != self.llm.tokenizer.pad_token_id
        return BatchEncoding(dict(
            input_ids = tokens,
            attention_mask = attention_mask,
        ))

    # (n, w) -> (n, w)
    def generate(self, query: BatchEncoding) -> LongTensor:
        generated = self.llm.model.generate(
            input_ids = query.input_ids,
            attention_mask = query.attention_mask,
            max_new_tokens = self.max_length,
            min_new_tokens = self.max_length,
            tokenizer = self.llm.tokenizer,
            do_sample = False,
            temperature = None,
            top_p = None,
            return_dict_in_generate = True,
            pad_token_id = self.llm.tokenizer.pad_token_id,
            eos_token_id = self.llm.tokenizer.eos_token_id,
            bos_token_id = self.llm.tokenizer.bos_token_id,
        )

        sequences = generated.sequences[:, -self.max_length:]
        ignores = torch.cumsum(torch.isin(sequences, self.stop_token_ids), dim = 1) > 0
        sequences[ignores] = self.llm.tokenizer.pad_token_id

        return sequences

    # (n, w0), (n, w1) -> (n)
    def probability(self, query: BatchEncoding, answer: LongTensor) -> list[float]:
        probs = self.batch_probability(query, self.batch_encode(answer))
        return probs.cpu().tolist()

    # (n, w0), (n, w1) -> (n)
    @torch.no_grad()
    def batch_probability(self, query: BatchEncoding, answer: BatchEncoding) -> FloatTensor:
        entropies = self.llm.logits(query, answer).log_softmax(dim = 2)
        probs = torch.where(
            answer.input_ids == self.llm.tokenizer.pad_token_id,
            torch.nan,
            entropies.gather(index = answer.input_ids.unsqueeze(2), dim = 2).squeeze(2),
        )

        return -torch.nanmean(probs, dim = 1)

    # (n, w) -> [n]
    def decode(self, tokens: LongTensor) -> list[str]:
        decoded = self.llm.tokenizer.batch_decode(
            tokens,
            skip_special_tokens = True,
            clean_up_tokenization_spaces = True,
        )
        return [x.strip() for x in decoded]

    @staticmethod
    def streq(a: str, b: str) -> bool:
        a = a.lower().replace('the', '').replace(',', '').strip()
        b = b.lower().replace('the', '').replace(',', '').strip()
        return a[:len(b)] == b[:len(a)]

    def answerCounterfactuals(self, questions: list[Object], counterfactuals: list[str], parametric: LongTensor, counterfactual: LongTensor) -> dict[str, Any]:
        output: dict[str, Any] = {}
        ctx_tokens = self.tokenise([
            q.format(prompt = self.llm.cf_prompt, context = context)
            for q, context in zip(questions, counterfactuals)
        ])

        contextual = self.generate(ctx_tokens)

        output['contextual'] = self.decode(contextual)
        output['ctx_proba'] = self.probability(ctx_tokens, contextual)

        output['ctx_param_proba'] = self.probability(ctx_tokens, parametric)
        output['ctx_cf_proba'] = self.probability(ctx_tokens, counterfactual)

        return output

    def answerChunk(self, questions: list[Object], use_counterfactuals: bool = True) -> dict[str, Any]:
        output: dict[str, Any] = {}

        base_tokens = self.tokenise([q.format(prompt = self.llm.prompt) for q in questions])
        parametric = self.generate(base_tokens)

        output['parametric'] = self.decode(parametric)
        output['base_proba'] = self.probability(base_tokens, parametric)

        flips = findFlips2(questions, output['parametric'])
        counterfactual = parametric[flips]

        output['counterfactual'] = self.decode(counterfactual)
        output['base_cf_proba'] = self.probability(base_tokens, counterfactual)

        output |= self.answerCounterfactuals(questions, output['counterfactual'], parametric, counterfactual)

        output['comparison'] = [
            'Parametric' if self.streq(a, p) else
            'Counterfactual' if self.streq(a, c) else
            'Other'
            for p, c, a in zip(output['parametric'], output['counterfactual'], output['contextual'])
        ]

        return output

    @torch.no_grad()
    def answerQueries(self, questions: list[Object]) -> dict[str, Any]:
        output: defaultdict[str, list[Any]] = defaultdict(lambda: [])

        chunks = chunkByQuestion(questions, max_batch_size = 120)
        logging.info(f'Answering {len(questions)} queries in {len(chunks)} chunks.')

        for e, chunk in enumerate(chunks, start = 1):
            logging.info(f'Parsing chunk ({e} / {len(chunks)}), which has size {len(chunk)}.', extra = {'rate_limit': 20})

            chunk_output = self.answerChunk(chunk)

            for k, v in chunk_output.items():
                output[k] += v

        return dict(output)
