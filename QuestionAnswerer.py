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
        # all_logits = torch.stack(outputs.logits, dim = 1)
        # text = self.llm.tokenizer.batch_decode(outputs.sequences)
        # es = [e for e, p in enumerate(text) if '760' in p]
        # if es == [31]:
        #     e = es[0]
        #     sequence = outputs.sequences[e]
        #     logit = all_logits[e]
        #     decoded = {x.cpu().item(): self.llm.tokenizer.decode(x) for x in sequence}
        #     ipdb.set_trace()

        return torch.stack(outputs.scores, dim = 1).softmax(dim = 2)

    # (n, w, v) -> (n, w); [(n, w), (n, w)]
    def winner(self, logits: FloatTensor) -> tuple[LongTensor, FloatTensor]:
        probs, path = logits.max(dim = 2)
        return path, probs

    # Decodes the tokens in `path`, and returns the average value within used tokens in `probs`.
    # (n, w); (n, w) -> [n]; [n]
    def decode(self, path: LongTensor, probs: FloatTensor) -> tuple[list[str], list[float]]:
        path = path.clone()
        probs = probs.clone()

        # left = torch.cumsum(~torch.isin(path, self.stop_token_ids), dim = 1) == 0
        left = torch.zeros_like(path, dtype = torch.bool)
        right = torch.cumsum(~left & torch.isin(path, self.stop_token_ids), dim = 1) > 0
        ignores = left | right

        path[ignores] = self.llm.tokenizer.pad_token_id
        probs[ignores] = torch.nan

        phrase = self.llm.tokenizer.batch_decode(path, skip_special_tokens = True, clean_up_tokenization_spaces = True)
        avg_probs = list(probs.nanmean(dim = 1).cpu().numpy())

        if any('.' in x or '\n' in x for x in phrase):
            bads = [e for e, x in enumerate(phrase) if '.' in x or '\n' in x]
            tokens = {x.cpu().item(): self.llm.tokenizer.decode(x.cpu().item()) for e in bads for x in path[e]}
            bad_tokens = {k: v for k, v in tokens.items() if k not in self.stop_token_ids and ('.' in v or '\n' in v)}
            raise ValueError(f'Unregistered tokens with stop characters generated: {bad_tokens}')

        return [x.strip() for x in phrase], avg_probs

    # Gets mean probabilities of logits along a sequence of paths.
    # (n, v); (n, w, v) -> [n]
    def gather(self, path: LongTensor, logits: FloatTensor) -> tuple[list[str], list[float]]:
        traces = logits.gather(index = path.unsqueeze(2), dim = 2).squeeze(2)
        return self.decode(path, traces)

    @staticmethod
    def streq(a: str, b: str) -> bool:
        a = a.lower().replace('the', '').replace(',', '').strip()
        b = b.lower().replace('the', '').replace(',', '').strip()
        return a[:len(b)] == b[:len(a)]

    def answerCounterfactuals(self, questions: list[Object], counterfactuals: list[str], param_path: LongTensor) -> dict[str, Any]:
        prompt = 'Answer the following question using the previous context in a few words and with no formatting.'

        output: dict[str, Any] = {}
        queries = [
            q.format(prompt = prompt, context = context)
            for q, context in zip(questions, counterfactuals)
        ]

        logits = self.query(queries)
        path, probs = self.winner(logits)
        output['contextual'], output['ctx_proba'] = self.decode(path, probs)

        _, output['ctx_param_proba'] = self.gather(param_path, logits)

        return output

    def answerChunk(self, questions: list[Object], counterfactual_flips = Optional[list[int]]) -> dict[str, Any]:
        prompt = 'Answer the following question in a few words and with no formatting.'

        output: dict[str, Any] = {}

        logits = self.query([q.format(prompt = prompt) for q in questions])
        path, probs = self.winner(logits)
        output['parametric'], output['base_proba'] = self.decode(path, probs)

        if counterfactual_flips is not None:
            cf_path = path[counterfactual_flips]
            counterfactuals, base_cf_mean_probs = self.gather(cf_path, logits)
            output['counterfactual'] = counterfactuals
            output['base_cf_proba'] = base_cf_mean_probs

            output |= self.answerCounterfactuals(questions, counterfactuals, param_path = path)

            output['comparison'] = [
                'Parametric' if self.streq(a, p) else
                'Counterfactual' if self.streq(a, c) else
                'Other'
                for p, c, a in zip(output['parametric'], output['counterfactual'], output['contextual'])
            ]

        return output

    def answerQueries(self, questions: list[Object]) -> dict[str, Any]:
        output: defaultdict[str, list[Any]] = defaultdict(lambda: [])

        chunks = chunkByQuestion(questions, max_batch_size = 100)
        logging.info(f'Answering {len(questions)} queries in {len(chunks)} chunks.')

        for e, chunk in enumerate(chunks, start = 1):
            logging.info(f'Parsing chunk ({e} / {len(chunks)}), which has size {len(chunk)}', extra = {'rate_limit': 20})

            flips = findFlips2(chunk)
            chunk_output = self.answerChunk(chunk, flips)

            for k, v in chunk_output.items():
                output[k] += v

        return dict(output)
