import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import logging
import math
import torch
import typing

from Models import Model
from typing import Optional, Union, Any
from Utils import Question, sample_counterfactual_flips, chunk_questions

from collections import defaultdict
from transformers import BatchEncoding

import ipdb

FloatTensor = torch.Tensor
LongTensor = torch.Tensor
BoolTensor = torch.Tensor

# A QuestionAnswerer is the main class to answer queries with a given model.
# Example Usage:
#   qa = QuestionAnswerer('llama', device = 'cuda', max_length = 20, max_batch_size = 75)
#   output = qa.answerQueries(Utils.combine_questions(base_questions, objects))
# The list of models can be found in `Model_dict` in `Models.py`.
class QuestionAnswerer:
    device: str
    max_length: int
    max_batch_size: int
    runs_per_question: int
    llm: Model

    def __init__(
        self,
        model: Union[str, Model],
        device: str = 'cpu',
        max_length: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        runs_per_question: Optional[int] = None,
    ):
        self.device = device
        self.max_length = max_length or 20
        self.max_batch_size = max_batch_size or 120
        self.runs_per_question = runs_per_question or 1

        if type(model) == str:
            model = Model.fromName(model, device = device)

        model = typing.cast(Model, model)
        self.llm = model

        # Generated list of stop tokens: period, newline, and various different end tokens.
        stop_tokens = {'.', '\n'}
        self.stop_token_ids = torch.tensor([
            v
            for k, v in self.llm.tokenizer.get_vocab().items()
            if
                k in ['<start_of_turn>', '<end_of_turn>', self.llm.tokenizer.special_tokens_map['eos_token']] or
                not stop_tokens.isdisjoint(self.llm.tokenizer.decode(v))
        ]).to(self.device)

    # Query data related to a list of questions, and return a dict with information about these runs.
    # Output elements:
    #  parametric: Parametric answer, as a string.
    #  base_proba: Perplexity of parametric answer in base query.
    #  counterfactual: Randomly selected counterfactual answer.
    #  base_cf_proba: Perplexity of counterfacutal answer in base query.
    #  contextual: Contextual answer, as a string.
    #  ctx_proba: Perplexity of contextual answer.
    #  ctx_param_proba: Perplexity of parametric answer when running contextual query.
    #  ctx_cf_proba: Perplexity of counterfactual answer when running contextual query.
    #  comparison: Comparison between parametric and contextual answer. Where does this answer come from?
    #  preference: Comparison between perplexity of paramertic and counterfactual answer on contextual query. Which one is the least surprising?
    def answerChunk(self, questions: list[Question]) -> dict[str, Any]:
        output: defaultdict[str, list[Any]] = defaultdict(lambda: [])

        # Get the tokens of the question and generate the parametric answer.
        base_tokens = self.tokenise([q.format(prompt = self.llm.prompt) for q in questions])
        parametric = self.generate(base_tokens)

        parametric_output = self.decode(parametric)
        base_proba_output = self.perplexity(base_tokens, parametric)

        # We possibly want several runs here, each with a different randomly sampled set of counterfactuals.
        for run in range(self.runs_per_question):
            run_output: dict[str, list[Any]] = dict(
                parametric = parametric_output,
                base_proba = base_proba_output,
            )

            # Sample the counterfactuals and add them to the output.
            run_output['question'] = questions
            flips = sample_counterfactual_flips(questions, run_output['parametric'])
            counterfactual = parametric[flips]

            run_output['counterfactual'] = self.decode(counterfactual)
            run_output['base_cf_proba'] = self.perplexity(base_tokens, counterfactual)

            # Answer the counterfactuals, and union this dictionary to the output dictionary.
            run_output |= self.answerCounterfactuals(questions, run_output['counterfactual'], parametric, counterfactual)

            # We want to compare each contextual answer to their parametric and counterfactual.
            run_output['comparison'] = [
                'Parametric' if self.streq(a, p) else
                'Contextual' if self.streq(a, c) else
                'Other'
                for p, c, a in zip(run_output['parametric'], run_output['counterfactual'], run_output['contextual'])
            ]

            # We also want to figure out if P_2 < P_3.
            run_output['preference'] = [
                'Parametric' if pp > cp else
                'Contextual'
                for pp, cp in zip(run_output['ctx_proba'], run_output['ctx_cf_proba'])
            ]

            for k, v in run_output.items():
                output[k].extend(v)

        return output

    # Given a list of questions with assigned counterfactuals, run contextual queries and return
    # a dictionary containing information about these runs.
    # Parameter list:
    #  questions: list of questions to ask.
    #  counterfactuals: counterfactual answers, as string.
    #  parametric: parametric answer, as set of tokens.
    #    This will be used to calculate the perplexity of this answer with the counterfactual context.
    #  counterfactual: counterfacutal answers, as a set of tokens.
    #    This is necessary since the same string might have several encodings, but we need exactly the same one generated by the model
    #    in the first place.
    def answerCounterfactuals(self, questions: list[Question], counterfactuals: list[str], parametric: LongTensor, counterfactual: LongTensor) -> dict[str, Any]:
        output: dict[str, Any] = {}
        ctx_tokens = self.tokenise([
            q.format(prompt = self.llm.cf_prompt, context = context)
            for q, context in zip(questions, counterfactuals)
        ])

        contextual = self.generate(ctx_tokens)

        output['contextual'] = self.decode(contextual)
        output['ctx_proba'] = self.perplexity(ctx_tokens, contextual)

        output['ctx_param_proba'] = self.perplexity(ctx_tokens, parametric)
        output['ctx_cf_proba'] = self.perplexity(ctx_tokens, counterfactual)

        output['context_attn'], output['question_attn'] = self.avgSelfAttentions(ctx_tokens)

        return output

    # Answer a list of Questions: run the queries, gather counterfactual values, run the queries
    # with counterfactual context, and return a `dict` with information to print.
    @torch.no_grad()
    def answerQueries(self, questions: list[Question]) -> dict[str, Any]:
        output: defaultdict[str, list[Any]] = defaultdict(lambda: [])

        chunks = chunk_questions(questions, max_batch_size = self.max_batch_size)
        logging.info(f'Answering {len(questions)} queries in {len(chunks)} chunks.')

        for e, chunk in enumerate(chunks, start = 1):
            logging.info(f'Parsing chunk ({e} / {len(chunks)}), which has size {len(chunk)}.', extra = {'rate_limit': 20})

            chunk_output = self.answerChunk(chunk)

            for k, v in chunk_output.items():
                output[k] += v

        return dict(output)

    def fakeTokens(self) -> BatchEncoding:
        return self.tokenise(['[Context: Montevideo is located in Egypt] Q: What country is Montevideo located in? A: Montevideo is located in'])

    # Gets the scaled mean self-attentions of the context section of a query and
    # of the section after the context.
    @torch.no_grad()
    def avgSelfAttentions(self, queries: BatchEncoding) -> tuple[list[float], list[float]]:
        attentions = self.llm.attentions(queries)
        attention_mean = attentions.mean(dim = (1, 2))
        diag = attention_mean.diagonal(dim1 = 1, dim2 = 2)
        scaled = (diag - diag.min(dim = 1, keepdims = True)[0]) / (diag.max(dim = 1, keepdims = True)[0] - diag.min(dim = 1, keepdims = True)[0])

        context_left = ((queries.input_ids == self.llm.tokenizer.pad_token_id) | (queries.input_ids == self.llm.tokenizer.eos_token_id))
        context_right = ((queries.input_ids == self.llm.tokenizer.convert_tokens_to_ids(']')) | (queries.input_ids == self.llm.tokenizer.convert_tokens_to_ids('].')))

        context_area = ~context_left & (context_right.cumsum(dim = 1) == 0)
        later_area = context_right.cumsum(dim = 1) > 0

        context = scaled.clone()
        context[~context_area] = torch.nan

        later = scaled.clone()
        later[~later_area] = torch.nan

        return context.nanmean(dim = 1).cpu().tolist(), later.nanmean(dim = 1).cpu().tolist()

    # Tokenise a list of phrases.
    # [n] -> (n, w)
    def tokenise(self, phrases: list[str]) -> BatchEncoding:
        return self.llm.tokenizer(
            phrases,
            return_tensors = 'pt',
            return_attention_mask = True,
            padding = True,
        ).to(self.device)

    # Generate an attention mask for a sequence of tokens.
    # (n, w) -> (n, w)
    def batch_encode(self, tokens: LongTensor) -> BatchEncoding:
        attention_mask = tokens != self.llm.tokenizer.pad_token_id
        return BatchEncoding(dict(
            input_ids = tokens,
            attention_mask = attention_mask,
        ))

    # Use Greedy decoding to generate an answer to a certain query.
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

        # Ensure that all the sequences only contain <PAD> after their first stop token.
        sequences = generated.sequences[:, -self.max_length:]
        ignores = torch.cumsum(torch.isin(sequences, self.stop_token_ids), dim = 1) > 0
        sequences[ignores] = self.llm.tokenizer.pad_token_id

        return sequences

    # Return the perplexity of a certain sequence of tokens being the answer to a
    # certain query, as a list of floats in CPU.
    # (n, w0), (n, w1) -> (n)
    def perplexity(self, query: BatchEncoding, answer: LongTensor) -> list[float]:
        probs = self.batch_perplexity(query, self.batch_encode(answer))
        return probs.cpu().tolist()

    # Return the perplexity of a certain sequence of tokens being the answer to a
    # certain query.
    # (n, w0), (n, w1) -> (n)
    @torch.no_grad()
    def batch_perplexity(self, query: BatchEncoding, answer: BatchEncoding) -> FloatTensor:
        entropies = self.llm.logits(query, answer).log_softmax(dim = 2)
        entropies /= math.log(2)
        probs = torch.where(
            answer.input_ids == self.llm.tokenizer.pad_token_id,
            torch.nan,
            entropies.gather(index = answer.input_ids.unsqueeze(2), dim = 2).squeeze(2),
        )

        return torch.pow(2, -torch.nanmean(probs, dim = 1))

    # Decode a sequence of tokens into a list of strings.
    # (n, w) -> [n]
    def decode(self, tokens: LongTensor) -> list[str]:
        decoded = self.llm.tokenizer.batch_decode(
            tokens,
            skip_special_tokens = True,
            clean_up_tokenization_spaces = True,
        )
        return [x.strip() for x in decoded]

    # Compare strings for equality to later check whether an answer is parametric or contextual.
    # For simplicity, we remove stop words and gather only the subset of words.
    @staticmethod
    def streq(a: str, b: str) -> bool:
        a = a.lower().replace('the', '').replace(',', '').strip()
        b = b.lower().replace('the', '').replace(',', '').strip()
        return a[:len(b)] == b[:len(a)]
