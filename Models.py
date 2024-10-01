import logging

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BatchEncoding
from torch import nn, tensor
from torch import FloatTensor, Tensor
import torch

# Dictionary of models, containing all of the models aliases and their respective models.
Model_dict = {
    'llama': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'llama-405b': 'meta-llama/Meta-Llama-3.1-405B-Instruct',
    'flan-t5': 'google/flan-t5-base',
    'flan-t5-small': 'google/flan-t5-small',
    'flan-t5-base': 'google/flan-t5-base',
    'flan-t5-large': 'google/flan-t5-large',
    'flan-t5-xl': 'google/flan-t5-xl',
    'flan-t5-xxl': 'google/flan-t5-xxl',
    'gemma': 'google/gemma-2-9b-it',
    'gemma-27b': 'google/gemma-2-27b-it',
    'falcon2': 'tiiuae/falcon-11b',
    'falcon-180b': 'tiiuae/falcon-180b-chat',
    'falcon-40b': 'tiiuae/falcon-40b-instruct',
    'falcon-7b': 'tiiuae/falcon-7b-instruct',
    'distilbert': 'distilbert/distilbert-base-uncased-distilled-squad',
    'roberta': 'FacebookAI/roberta-base',
    'roberta-large': 'FacebookAI/roberta-large',
    'roberta-squad': 'deepset/roberta-base-squad2',
    'mixtral': 'mistralai/Mixtral-8x22B-Instruct-v0.1',
    'dummy': '',
}

# Virtual class containing a model.
# Derived classes should reimplement __init__ and logits, and attention.
# They should also save their tokeniser and HF model.
class Model(nn.Module):
    name: str
    model_name: str
    device: str

    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM

    # Construct a model from a certain name.
    # This should be the main constructor of models.
    @staticmethod
    def fromName(name: str, device: str = 'cpu') -> 'Model':
        if name == 'dummy':
            return DummyModel()

        if name in ('llama-70b', 'gemma-27b'):
            return LargeDecoderOnlyModel(name, device)

        if 't5' in name:
            return Seq2SeqModel(name, device)

        return DecoderOnlyModel(name, device)

    def __init__(self, name: str, device: str = 'cuda'):
        super().__init__()
        self.name = name
        self.model_name = Model_dict[name]
        self.device = device

    @torch.no_grad()
    def logits(self, query: BatchEncoding, answer: BatchEncoding) -> FloatTensor:
        raise NotImplementedError('logits called from generic Model class')

    @torch.no_grad()
    def attentions(self, query: BatchEncoding) -> FloatTensor:
        raise NotImplementedError('attentions called from generic Model class')

# Decoder-only model, such as llama, use AutoModelForCausalLM.
class DecoderOnlyModel(Model):
    def __init__(self, name: str, device: str = 'cuda'):
        super().__init__(name, device)

        self.prompt = ''
        self.cf_prompt = ''

        kwargs = {}
        # Some models require special configuration.
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

    # Decoder-only generation with teacher forcing: calculate the next token given the previous forced tokens.
    @torch.no_grad()
    def logits(self, query: BatchEncoding, answer: BatchEncoding) -> FloatTensor:
        w0 = query.input_ids.shape[1]
        w1 = answer.input_ids.shape[1]

        input_ids = torch.cat([query.input_ids, answer.input_ids], dim = 1)
        attention_mask = torch.cat([query.attention_mask, answer.attention_mask], dim = 1)

        return self.model(input_ids, attention_mask = attention_mask).logits[:, w0 - 1 : w0 + w1 - 1]

    @torch.no_grad()
    def attentions(self, queries: BatchEncoding) -> FloatTensor:
        outputs = self.model(**queries, output_attentions = True).attentions
        return torch.stack(outputs, dim = 1)

# Seq2Seq model, such as Flan-T5.
class Seq2SeqModel(Model):
    def __init__(self, name: str, device: str = 'cpu'):
        super().__init__(name, device)

        self.prompt = ''
        self.cf_prompt = ''

        kwargs = dict(
            padding_side = 'right',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            clean_up_tokenization_spaces = True,
            **kwargs,
        )

        logging.info(f'Loading Seq2Seq model for {self.model_name} using {torch.cuda.device_count()} GPUs.')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            device_map = 'auto' if self.device == 'cuda' else self.device,
            torch_dtype = torch.bfloat16,
            pad_token_id = self.tokenizer.pad_token_id,
            bos_token_id = self.tokenizer.bos_token_id,
            eos_token_id = self.tokenizer.eos_token_id,
            low_cpu_mem_usage = True,
        )
        self.model.eval()

    @staticmethod
    def pad(tensor: Tensor, length: int, value) -> Tensor:
        right = torch.full((tensor.shape[0], length - tensor.shape[1]), value)
        return torch.cat([tensor, right.to(tensor.device)], dim = 1)

    @torch.no_grad()
    def logits(self, query: BatchEncoding, answer: BatchEncoding) -> FloatTensor:
        length = max(query.input_ids.shape[1], answer.input_ids.shape[1])

        input_ids = self.pad(query.input_ids, length, self.tokenizer.pad_token_id)
        attention_mask = self.pad(query.attention_mask, length, 0)
        decoder_input_ids = self.pad(self.model._shift_right(answer.input_ids), length, self.tokenizer.pad_token_id)

        return self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_input_ids = decoder_input_ids,
        ).logits[:, : answer.input_ids.shape[1]]

    @torch.no_grad()
    def attentions(self, queries: BatchEncoding) -> FloatTensor:
        outputs = self.model(
            **queries,
            output_attentions = True,
            decoder_input_ids = torch.full_like(queries.input_ids, self.tokenizer.pad_token_id).to(self.device),
        ).decoder_attentions
        return torch.stack(outputs, dim = 1)

# Large decoder-only model.
# Similar to DecoderOnlyModel, but eagerly deletes the model when the class is deleted.
# Assumes you need 2 GPUs to run this.
class LargeDecoderOnlyModel(DecoderOnlyModel):
    def __init__(self, name, device: str = 'cuda'):
        if torch.cuda.device_count() < 2:
            raise ValueError(f'At least two GPUs are needed to run {name}')

        super().__init__(name, device)

    def __del__(self):
        logging.info(f'Deleting large model {self.name}')
        del self.model
        torch.cuda.empty_cache()

# Dummy model, used for testing.
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

# If called separately, just print the names of the models.
def main():
    print(f'{"Model Name":>15} | {"Huggingface Model":<40}')
    print((15 + 1) * '-' + '|' + (40 + 1) * '-')
    for name, model in Model_dict.items():
        print(f'{name:>15} | {model:<40}')

if __name__ == '__main__':
    main()
