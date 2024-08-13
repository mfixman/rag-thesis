import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from transformers import *
import logging
import torch
from torch import nn

Model_dict = {
    'llama': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'falcon2': 'tiiuae/falcon-11b',
    'llama-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'falcon-180b': 'tiiuae/falcon-180b-chat',
    'falcon-40b': 'tiiuae/falcon-40b-instruct',
    'falcon-7b': 'tiiuae/falcon-7b-instruct',
    'distilbert': 'distilbert/distilbert-base-uncased-distilled-squad',
    'roberta': 'FacebookAI/roberta-base',
    'roberta-large': 'FacebookAI/roberta-large',
    'roberta-squad': 'deepset/roberta-base-squad2',
    'llama-405b': 'meta-llama/Meta-Llama-3.1-405B-Instruct',
    'gemma': 'google/gemma-2-9b-it',
    'mixtral': 'mistralai/Mixtral-8x22B-Instruct-v0.1',
}

class Model(nn.Module):
    name: str
    model_name: str
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.model_name = Model_dict[name]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )
        self.model = self.getModel(self.model_name)
        self.model.eval()

    def getModel(self, model_name: str) -> AutoModelForCausalLM:
        logging.info(f'Getting {model_name}')
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map = 'auto',
            )
        except OSError:
            pass

        for a in range(2, 10):
            logging.info(f'Attempt {a}/10 for {model_name}')
            try:
                return AutoModelForCausalLM.from_pretrained(
                    model_name,
                    force_download = True,
                    resume_download = False,
                )
            except OSError:
                pass

        logging.error('Failed 10 attempts for {model_name}. Giving up.')
        raise

class DummyModel(Model):
    def __init__(self):
        self.name = 'dummy'
        self.tokenizer = self.DummyTorchTokenizer()
        self.model = self.DummyTorchmodel()

    class DummyTorchTokenizer:
        def __call__(self, *args, **kwargs):
            return [0, 1, 2, 3]

        def decode(self, *args, **kwargs):
            return 'Dummy text.'

    class DummyTorchModel:
        def generate(self, *args, **kwargs):
            return None

class QuestionAnswerer:
    device: str
    llms: list[Model]

    def __init__(self, model_names: list[str], device: str = 'cpu'):
        self.device = device
        
        assert all(x in Model_dict for x in model_names)
        self.llms = [Model(x).to(device) for x in model_names]

    @torch.no_grad()
    def query(self, question: str, max_length: int, short: bool = False, return_rest: bool= False) -> list[str] | list[tuple[str, dict[str, float]]]
        answers = {}
        rest = {}
        for llm in self.llms:
            inputs = llm.tokenizer(question, return_tensors = "pt", truncation = True)
            outputs = llm.model.generate(
                input_ids = inputs["input_ids"].to(self.device),
                attention_mask = inputs['attention_mask'].to(self.device),
                max_new_tokens = max_length,
                stop_strings = '.',
                do_sample = False,
                tokenizer = llm.tokenizer,
                output_logits = True,
                return_dict_in_generate = True,
                temperature = None,
            )

            answer = llm.tokenizer.decode(outputs.sequences[0], skip_special_tokens = True)
            if short:
                answer = answer.removeprefix(question).strip('"[]. \n')

            answers[llm.name] = answer
            rest[llm.name] = dict(
                logit_min = torch.tensor([x.squeeze(0).max() for x in outputs.logits]).min().item(),
                logit_prod = torch.tensor([torch.nn.functional.softmax(x.squeeze(0), dim = 0).max() for x in outputs.logits]).prod().item(),
            )

        if not return_rest:
            return answers

        return answers, rest
