import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import http
import os
import sys
import faiss
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from transformers import DPRQuestionEncoderTokenizer, BartTokenizer
from transformers import AutoTokenizer, AutoModel

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

logging.basicConfig(
    format='[%(asctime)s] %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

def RAGRetrieverRetry(name: str, index_name: str, retries: int = 1, **kwargs) -> RagRetriever:
    proxies = dict(
        http = 'http://hpc-proxy00.city.ac.uk:3128',
        https = 'https://hpc-proxy00.city.ac.uk:3128',
    )

    for attempts in range(1, retries + 1):
        logging.info(f'Starting attempt {attempts}/{retries}.')
        try:
            retriever = RagRetriever.from_pretrained(
                name,
                index_name = index_name,
                proxies = proxies,
                **kwargs,
            )
            logging.info('Downloaded correctly :-D')
            return retriever
        except (http.client.IncompleteRead) as e:
            logging.error(f'Incomplete Read: {e}')
            logging.info('Restarting ot continuing')

    logging.error('Failing forever.')
    raise

def main():
	logging.info('Starting')
	rag = 'facebook/rag-token-nq'

	logging.info('Getting tokenizer')
	tokenizer = RagTokenizer.from_pretrained(rag)

	logging.info('Getting retriever')
	retriever = RAGRetrieverRetry(
		rag,
		index_name = 'wiki_dpr',
        retries = 5,
		use_dummy_dataset = False,
        trust_remote_code = True,
	)

	logging.info('Getting RAG token for generation')
	model = RagTokenForGeneration.from_pretrained(
		rag,
		retriever = retriever,
		ignore_mismatched_sizes = True,
	)

	try:
		question = sys.argv[1]
	except IndexError:
		question = 'who holds the record in 100m freestyle'

	logging.info('Prepare seq2seq batch')
	input_dict = tokenizer.prepare_seq2seq_batch(
		question,
		return_tensors = 'pt',
	)

	logging.info('Generating model')
	generated = model.generate(input_ids = input_dict['input_ids']) 

	logging.info('Batch decoding')
	print(tokenizer.batch_decode(generated, skip_special_tokens = True)[0]) 

if __name__ == '__main__':
	main()
