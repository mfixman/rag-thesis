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

        outputs = self.llm.model.generate(
            input_ids = tokens.input_ids,
            attention_mask = tokens.attention_mask,
            max_new_tokens = self.max_length,
            min_new_tokens = self.max_length - 1,
            # stop_strings = ['.', '\n'],
            do_sample = False,
            tokenizer = self.llm.tokenizer,
            output_logits = True,
            output_scores = True,
            return_dict_in_generate = True,
            temperature = None,
            top_p = None,
            pad_token_id = self.llm.tokenizer.pad_token_id,
            eos_token_id = self.llm.tokenizer.eos_token_id,
            bos_token_id = self.llm.tokenizer.bos_token_id,
        )

        return torch.stack(outputs.scores, dim = 1).softmax(dim = 2)

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

        sequences = generated.sequences[:, query.input_ids.shape[1]:]
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
        w0 = query.input_ids.shape[1]
        w1 = answer.input_ids.shape[1]

        input_ids = torch.cat([query.input_ids, answer.input_ids], dim = 1)
        attention_mask = torch.cat([query.attention_mask, answer.attention_mask], dim = 1)

        logits = self.llm.model(input_ids, attention_mask = attention_mask).logits[:, w0 - 1 : w0 + w1 - 1]
        entropies = logits.log_softmax(dim = 2)
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

    # (n, w, v) -> (n, w); [(n, w), (n, w)]
    def winner(self, logits: FloatTensor) -> tuple[LongTensor, FloatTensor]:
        probs, path = logits.max(dim = 2)
        return path, probs

    # Decodes the tokens in `path`, and returns the average value within used tokens in `probs`.
    # (n, w); (n, w) -> [n]; [n]
    def decode_2(self, path: LongTensor, probs: FloatTensor) -> tuple[list[str], list[float]]:
        # path = path.clone()
        # probs = probs.clone()

        # left = torch.cumsum(~torch.isin(path, self.stop_token_ids), dim = 1) == 0
        left = torch.zeros_like(path, dtype = torch.bool)
        right = torch.cumsum(~left & torch.isin(path, self.stop_token_ids), dim = 1) > 0
        ignores = left | right

        path[ignores] = self.llm.tokenizer.pad_token_id
        probs[path == self.llm.tokenizer.pad_token_id] = torch.nan

        phrase = self.llm.tokenizer.batch_decode(path, skip_special_tokens = True, clean_up_tokenization_spaces = True)
        perplexity = torch.exp(-torch.nanmean(torch.log(probs), dim = 1))

        if any('.' in x or '\n' in x for x in phrase):
            bads = [e for e, x in enumerate(phrase) if '.' in x or '\n' in x]
            tokens = {x.cpu().item(): self.llm.tokenizer.decode(x.cpu().item()) for e in bads for x in path[e]}
            bad_tokens = {k: v for k, v in tokens.items() if k not in self.stop_token_ids and ('.' in v or '\n' in v)}
            raise ValueError(f'Unregistered tokens with stop characters generated: {bad_tokens}')

        if torch.any(perplexity < 1):
            logging.warn(f'{torch.sum(perplexity < 1)} entries found with perplexity less than 1!')

        return [x.strip() for x in phrase], perplexity.tolist()

    # Gets mean probabilities of logits along a sequence of paths.
    # (n, v); (n, w, v) -> [n]
    def gather(self, path: LongTensor, logits: FloatTensor) -> tuple[list[str], list[float]]:
        traces = logits.gather(index = path.unsqueeze(2), dim = 2).squeeze(2)
        return self.decode_2(path, traces)

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
