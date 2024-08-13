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

            answers[llm.name] = answer
            rest[llm.name] = dict(
                logit_min = torch.tensor([x.squeeze(0).max() for x in outputs.logits]).min().item(),
                logit_prod = torch.tensor([torch.nn.functional.softmax(x.squeeze(0), dim = 0).max() for x in outputs.logits]).prod().item(),
            )

        if not self.return_rest:
            return answers

        return answers, rest
