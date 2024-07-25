from transformers import *
import logging
import torch

class RAG:
    def __init__(self, rag_name: str = 'facebook/rag-token-base', device: str = 'cpu'):
        logging.info('Getting RAG tokenizer')
        self.tokenizer = RagTokenizer.from_pretrained(rag_name)

        logging.info('Getting RAG retriever')
        self.retriever = RagRetriever.from_pretrained(
            rag_name,
            index_name = 'custom_index',
        )

        logging.info('Getting RAG tokenizer for generation')
        self.model = RagTokenizerForGeneration.from_pretrained(
            rag_name,
            retriever = self.retriever,
        )
        self.model.eval()

        self.device = device

    @torch.no_grad()
    def retrieve_context(self, question):
        logging.info('Tokenizing RAG')
        input_ids = self.tokenizer(question, return_tensors = 'pt').input_ids.to(self.device)

        logging.info('Retrieving RAG')
        documents = self.model.to(self.device).retriever.retrieve(input_ids)
        return '; '.join([d['text'] for d in documents])

class EmptyRAG(RAG):
    def __init__(self, rag_name: str, device: str):
        pass

    def retrieve_context(self, question):
        return ''
