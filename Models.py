import logging

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BatchEncoding
from torch import nn, tensor
from torch import FloatTensor, LongTensor, BoolTensor, Tensor
import torch
import ipdb

import sys
import os

sys.path.append('./atlas/')
from atlas.src.atlas import Atlas
from atlas.src.model_io import load_atlas_model
from atlas.src.options import Options

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
    'atlas': 'atlas/data/models/atlas_nq/xl',
    'dummy': '',
}

class Model(nn.Module):
    name: str
    model_name: str
    device: str

    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM

    @staticmethod
    def fromName(name: str, device: str = 'cpu') -> 'Model':
        if name == 'dummy':
            return DummyModel()

        if name in ('llama-70b', 'gemma-27b'):
            return LargeDecoderOnlyModel(name, device)

        if 't5' in name:
            return Seq2SeqModel(name, device)

        if name in ('atlas'):
            return RetrievalAugmentedModel(name, device)

        return DecoderOnlyModel(name, device)

    def __init__(self, name: str, device: str = 'cuda'):
        super().__init__()
        self.name = name
        self.model_name = Model_dict[name]
        self.device = device

    # [n] -> (n, w)
    def tokenise(self, phrases: list[str]) -> BatchEncoding:
        return self.tokenizer(
            phrases,
            return_tensors = 'pt',
            return_attention_mask = True,
            padding = True,
        ).to(self.device)

class DecoderOnlyModel(Model):
    def __init__(self, name: str, device: str = 'cuda'):
        super().__init__(name, device)

        # self.prompt = 'Answer the following question in a few words and with no formatting.'
        # self.cf_prompt = 'Answer the following question using the previous context in a few words and with no formatting.'
        self.prompt = ''
        self.cf_prompt = ''

        kwargs = {}
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

    @torch.no_grad()
    def logits(self, query: BatchEncoding, answer: BatchEncoding) -> FloatTensor:
        w0 = query.input_ids.shape[1]
        w1 = answer.input_ids.shape[1]

        input_ids = torch.cat([query.input_ids, answer.input_ids], dim = 1)
        attention_mask = torch.cat([query.attention_mask, answer.attention_mask], dim = 1)

        return self.model(input_ids, attention_mask = attention_mask).logits[:, w0 - 1 : w0 + w1 - 1]

class Seq2SeqModel(Model):
    def __init__(self, name: str, device: str = 'cpu'):
        super().__init__(name, device)

        # self.prompt = 'Answer the following question in a few words, and write a period at the end of the answer.'
        # self.cf_prompt = 'Answer the following question in a few words using the previous context, and write a period at the end of the answer.'
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

class LargeDecoderOnlyModel(DecoderOnlyModel):
    def __init__(self, name, device: str = 'cpu'):
        if torch.cuda.device_count() < 2:
            raise ValueError(f'At least two GPUs are needed to run {name}')

        super().__init__(name, device)

    def __del__(self):
        logging.info(f'Deleting large model {self.name}')
        del self.model
        torch.cuda.empty_cache()

class RetrievalAugmentedModel(Seq2SeqModel):
    @staticmethod
    def atlas_options(size: str):
        return Options.from_dict(
            index_mode="flat",
            save_index_n_shards=128,
            qa_prompt_format="question: {question} answer: <extra_id_0>",
            max_lm_context_ratio=0.5,
            min_lm_context_ratio=0.5,
            mlm_mean_noise_span_length=3,
            mlm_noise_density=0.15,
            retriever_format="{title} {text}",
            encoder_format="{query} title: {title} context: {text}",
            n_to_rerank_with_retrieve_with_rerank=128,
            freeze_retriever_steps=-1,
            filtering_overretrieve_ratio=2,
            temperature_gold=0.01,
            temperature_score=0.01,
            gold_score_mode="ppmean",
            retriever_n_context=5,
            retriever_model_path="facebook/contriever",
            max_passages=-1,
            n_context=1,
            text_maxlength=200,
            precision="fp32",
            refresh_index="-1",
            beta2=0.999,
            alpha=1.0,
            epsilon=1e-6,
            weight_decay=0.1,
            scheduler="cosine",
            clip=1.0,
            lr=1e-4,
            dropout=0.1,
            accumulation_steps=1,
            total_steps=1000,
            warmup_steps=1000,
            eval_data=[],
            train_data=[],
            per_gpu_embedder_batch_size=16 * 512,
            per_gpu_batch_size=1,
            checkpoint_dir="experiments/",
            name="another_experiment",
            # model_path = f'data/models/atlas_nq/{size}',
            # load_index_path = f'data/indices/atlas_nq/wiki/{size}',
            reader_model_type = f'google/t5-{size}-lm-adapt',
        )

    class DummyAtlas(Atlas):
        def __init__(self):
            pass

        def generate(self, input_ids, attention_mask, **kwargs):
            logging.info('Called dummy generate')
            outputs = []
            for ids, mask in zip(input_ids, attention_mask):
                outputs.append(self.reader.generate(
                    input_ids = ids,
                    attention_mask = mask,
                    num_return_sequences=1,
                    **kwargs,
                ))

            return torch.cat(outputs, axis = 0)

    def __init__(self, name: str, device: str):
        Model.__init__(self, name, device)

        size = Model_dict[name].split('/')[-1]
        opt = self.atlas_options(size)
        self.atlas, _, _, _, _, _, _ = load_atlas_model(
            Model_dict[name],
            opt,
            reset_params = True,
            eval_only = True,
        )

        self.tokenizer = self.atlas.reader_tokenizer
        self.model = self.DummyAtlas()
        self.model.__dict__.update(self.atlas.__dict__)

        self.prompt = ''
        self.cf_prompt = ''

    # [n] -> (n, w)
    def tokenise(self, phrases: list[str]) -> BatchEncoding:
        return self.tokenizer(
            phrases,
            return_tensors = 'pt',
            return_attention_mask = True,
            padding = True,
            # padding = 'max_length',
            # max_length = 100,
        ).to(self.device)

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

def main():
    print(f'{"Model Name":>15} | {"Huggingface Model":<40}')
    print((15 + 1) * '-' + '|' + (40 + 1) * '-')
    for name, model in Model_dict.items():
        print(f'{name:>15} | {model:<40}')

if __name__ == '__main__':
    main()
