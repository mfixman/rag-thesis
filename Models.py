import logging

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn, tensor
import torch

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
    'gemma-27b': 'google/gemma-2-27b-it',
    'mixtral': 'mistralai/Mixtral-8x22B-Instruct-v0.1',
    'dummy': '',
}

class Model(nn.Module):
    name: str
    model_name: str
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: str

    def __init__(self, name: str, device: str = 'cpu'):
        super().__init__()
        self.name = name
        self.model_name = Model_dict[name]
        self.device = device

        if 'llama' in name:
            kwargs = dict(
                pad_token = '<|reserved_special_token_0|>',
                padding_side = 'left',
            )
        elif 'gemma' in name:
            kwargs = dict(
                padding_side = 'right',
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            clean_up_tokenization_spaces = True,
            **kwargs,
        )

        logging.info(f'Loading model for {self.model_name} using {torch.cuda.device_count()} GPUs.')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map = 'auto' if self.device == 'cuda' else self.device,
            torch_dtype = torch.bfloat16,
            pad_token_id = self.tokenizer.pad_token_id,
            bos_token_id = self.tokenizer.bos_token_id,
            eos_token_id = self.tokenizer.eos_token_id,
            low_cpu_mem_usage = True,
        )
        self.model.eval()

        self.batch_size = 15000

        logging.info('All loaded!')

    @staticmethod
    def fromName(name: str, device: str = 'cpu') -> 'Model':
        if name == 'dummy':
            return DummyModel()

        if name in ('llama-70b', 'gemma-27b'):
            return LargeModel(name, device)

        return Model(name, device)

    def getModel(self, model_name: str) -> AutoModelForCausalLM:
        logging.info(f'Getting {model_name}')
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map = 'auto' if self.device == 'cuda' else self.device,
                torch_dtype = torch.float16,
                pad_token_id = self.tokenizer.pad_token_id,
                bos_token_id = self.tokenizer.bos_token_id,
                eos_token_id = self.tokenizer.eos_token_id,
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
                    device_map = self.device,
                )
            except OSError:
                pass

        logging.error('Failed 10 attempts for {model_name}. Giving up.')
        raise

    def __del__(self):
        logging.info(f'Deleting large model {self.name}')
        del self.model
        torch.cuda.empty_cache()


class LargeModel(Model):
    def __init__(self, name, device: str = 'cpu'):
        if torch.cuda.device_count() < 2:
            raise ValueError(f'At least two GPUs are needed to run {name}')

        super().__init__(name, device)
        self.batch_size = 5000

class DummyModel(Model):
    def __init__(self):
        nn.Module.__init__(self)
        self.name = 'dummy'
        self.tokenizer = self
        self.model = self
        self.sequences = ['dummy']
        self.logits = tensor([[[1., 2., 3.]]])

        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

    def to(self, *args, **kwargs):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def generate(self, *args, **kwargs):
        return self

    def __getitem__(self, key):
        return self

    def decode(self, *args, **kwargs):
        return 'Dummy text'

    def batch_decode(self, *args, **kwargs):
        return ['Dummy Text 1', 'Dummy Text 2']

    def shape(self):
        return (1, 2, 3)
