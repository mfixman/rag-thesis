import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.basicConfig(
    format='[%(asctime)s] %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = 'tiiuae/falcon-7b-instruct'
    
    logging.info(f'Getting pretrained model {model_name}')
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')

    logging.info(f'Getting pretrained tokenizer {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name).to('cuda')

    logging.info('Tokenizing input')
    inputs = tokenizer.encode('Where was Emmanuel Macron born?', return_tensors = 'pt').to('cuda')

    logging.info('Generating output')
    outputs = model.generate(inputs, max_length = 300, num_return_sequences = 1)

    logging.info('Decoding result')
    result = tokenizer.decode(outputs[0], skip_special_tokens = True)

    print(result)

if __name__ == '__main__':
    main()
