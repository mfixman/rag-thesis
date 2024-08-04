from transformers import *
import logging
import torch

class RAG:
    def __init__(self, rag_name: str = 'facebook/rag-sequence-nq', device: str = 'cpu', dummy = False):
        if dummy:
            logging.warn('Warning: Using dummy dataset')

        logging.info('Getting RAG config')
        self.config = RagConfig.from_pretrained(rag_name)

        logging.info('Getting RAG tokenizers')
        self.tokenizer = RagTokenizer.from_pretrained(rag_name)

        logging.info('Getting RAG retriever')
        self.retriever = RagRetriever.from_pretrained(
            rag_name,
            # config = self.config,
            # question_encoder_tokenizer = self.tokenizer,
            # generator_tokenizer = self.tokenizer,
            index_name = 'compressed' if not dummy else 'exact',
            use_dummy_dataset = dummy,
        )

        logging.info('Getting RAG model')
        self.model = RagSequenceForGeneration.from_pretrained(
            rag_name,
            retriever = self.retriever,
        ).to(device)
        self.model.eval()

        self.device = device

    @torch.no_grad()
    def retrieve_context(self, question):
        logging.info('Preparing RAG batch')
        input_dict = self.tokenizer.prepare_seq2seq_batch(question, return_tensors = 'pt')

        logging.info('Generating answer')
        generated = self.model.generate(input_ids = input_dict['input_ids'])

        logging.info('Decoding answer')
        answer = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens = True
        )
        return answer[0]
        # logging.info('Tokenizing RAG')
        # input_ids = self.tokenizer(question, return_tensors = 'pt').input_ids.to(self.device)

        # logging.info('Retrieving outputs')
        # datas = input_ids.cpu().detach().numpy()
        # logging.info(f'Datas shape: {datas.shape}')
        # outputs = self.retriever.retrieve(datas, n_docs = 1)

        # logging.info('Retrieving contexts')
        # retrieved_contexts = [self.tokenizer.decode(doc, skip_special_tokens = True) for doc in outputs['context_input_ids'][0]]
        # return '; '.join(retrieved_contexts)

        # logging.info('Retrieving RAG')
        # documents = self.model.to(self.device).retriever.retrieve(input_ids)
        # return '; '.join([d['text'] for d in documents])

class ConstRAG(RAG):
    def __init__(self, const, **kwargs):
        self.const = const

    def retrieve_context(self, question):
        return self.const

class EmptyRAG(RAG):
    def __init__(self, *args, **kwargs):
        pass

    def retrieve_context(self, question):
        return ''
