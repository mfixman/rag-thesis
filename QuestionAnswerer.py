import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

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
        models: list[Model],
        device: str = 'cpu',
        max_length: Optional[int] = None,
        short: bool = True,
        return_rest: bool = True
    ):
        self.device = device
        self.max_length = max_length
        self.short = short
        self.return_rest = return_rest

        self.llms = [x.to(device) for x in models]

    @torch.no_grad()
    def query(self, question: str) -> Union[dict[str, str], tuple[dict[str, str], dict[str, dict[str, float]]]]:
        answers = {}
        rest = {}
        for llm in self.llms:
            inputs = llm.tokenizer(question, return_tensors = "pt", truncation = True)
            outputs = llm.model.generate(
                input_ids = inputs['input_ids'].to(self.device),
                attention_mask = inputs['attention_mask'].to(self.device),
                max_new_tokens = self.max_length,
                stop_strings = '.',
                do_sample = False,
                tokenizer = llm.tokenizer,
                output_logits = True,
                return_dict_in_generate = True,
                temperature = None,
                top_p = None,
            )

            answer = llm.tokenizer.decode(outputs.sequences[0], skip_special_tokens = True)
            if self.short:
                answer = answer.removeprefix(question).strip('"[]. \n')

        generated = outputs.sequences[:, tokens['input_ids'].shape[1]:]
        bad_tokens = torch.tensor(self.llm.tokenizer.convert_tokens_to_ids(['.'] + list(itertools.chain.from_iterable(self.llm.tokenizer.special_tokens_map.values())))).to(self.device)
        invalid_mask = torch.isin(generated, bad_tokens)

        generated = outputs.sequences

        answers = [
            self.llm.tokenizer.decode(
                x,
                skip_special_tokens = True,
            ).strip('.\n ' + self.llm.tokenizer.pad_token)
            for x in generated
        ]

        return answers, rest
