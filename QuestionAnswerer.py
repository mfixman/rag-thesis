import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import itertools
import logging
import torch
import typing

from torch import LongTensor, FloatTensor
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

from Models import Model
from typing import Optional, Union

import ipdb

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

    @torch.no_grad()
    def query(self, questions: list[str]) -> tuple[list[str], FloatTensor]:
        tokens = self.llm.tokenizer(
            questions,
            return_tensors = 'pt',
            return_attention_mask = True,
            padding = True,
        ).to(self.device)

        batch_size = 15000
        chunks = 1 + (tokens['input_ids'].shape[0] * tokens['input_ids'].shape[1]) // batch_size
        input_ids = tokens['input_ids'].chunk(chunks, dim = 0)
        attention_masks = tokens['attention_mask'].chunk(chunks, dim = 0)

        logging.info(f"Answering questions for {len(questions)} × {tokens['input_ids'].shape[1]} = {len(questions) * tokens['input_ids'].shape[1]} in {chunks} chunks.")
        sequences: list[str] = []
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

            sequences.extend(
                self.llm.tokenizer.batch_decode(
                    outputs.sequences,
                    skip_special_tokens = True,
                    clean_up_tokenization_spaces = True,
                )
            )
            all_logits.append(torch.stack(outputs.logits, dim = 1).softmax(dim = 2))

        answers = [a.removeprefix(q).split('\n')[0].split('.')[0] for q, a in zip(questions, sequences)]
        return answers, torch.cat(all_logits, dim = 0)

    def gather(self, logits: FloatTensor, words: list[str]) -> list[float]:
        assert logits.shape[0] == len(words)
        assert torch.all((logits >= 0) & (logits < 1))
        assert torch.isclose(logits.sum(dim = 2), torch.ones(logits.shape[0:2]).to(self.device)).all()

        ids = self.llm.tokenizer(
            words,
            return_tensors = 'pt',
            padding = True,
        ).input_ids.to(self.device)
        logging.info(ids.shape)

        assert logits.shape[1] >= ids.shape[1]
        logits = logits[:, :ids.shape[1], :]

        traces = logits.gather(index = ids.unsqueeze(2), dim = 2).squeeze(2)
        traces[ids == self.llm.tokenizer.pad_token_id] = torch.nan
        logging.info(list(zip(ids, traces)))

        return traces.nanmean(dim = 1).cpu().numpy()
