import logging

from transformers import AutoTokenizer, AutoModelForCausalLM
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

    @staticmethod
    def fromNames(names: list[str]) -> list[Model]:
        return [DummyModel() if x == 'dummy' else Model(x) for x in names]

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
