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
}

class Model(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.model_name = Model_dict[name]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )
        self.model = self.getModel(self.model_name)
        self.model.eval()

    def getModel(self, model_name):
        logging.info(f'Getting {model_name}')
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_name,
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

class QuestionAnswerer:
    def __init__(self, model_names, device = 'cpu'):
        self.device = device
        
        assert all(x in Model_dict for x in model_names)
        self.llms = [Model(x).to(device) for x in model_names]

    @torch.no_grad()
    def query(self, question, max_length, short = False):
        answers = {}
        for llm in self.llms:
            # logging.info('Tokenising')
            inputs = llm.tokenizer(question, return_tensors = "pt", truncation = True)

            outputs = llm.model.generate(
                inputs["input_ids"].to(self.device),
                max_new_tokens = len(question) + max_length,
                num_return_sequences = 1,
                pad_token_id = llm.tokenizer.eos_token_id,
                attention_mask = inputs['attention_mask'].to(self.device),
                do_sample = False,
                early_stopping = True,
                num_beams = 3,
            )

            # logging.info('Decoding')
            answer = llm.tokenizer.decode(outputs[0], skip_special_tokens = True)
            if short:
                answer = answer.splitlines()[0].split('.')[0] + '.'

            answers[llm.name] = answer

        return answers