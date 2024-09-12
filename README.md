# Enhancing Knowledge Grounding in Retrieval-Augmented Languages Models
An ongoing MSc thesis project by Martin Fixman, supervised by Tillman Weyde.

## What is this about (for now)?
We are competing the grounding of different types of knowledge of many LLMs on several RAG models containing counterfactual information.

Better description to be completer later!

![Data preparation pipeline](figures/Figure_1.png)

![Model evaluation](figures/Figure_2.png)

## Recommendations
Many of the models are large, and it might be useful to download them using the Huggingface CLI first.

For example,
```
$ huggingface-cli download --repo-type model 'facebook/rag-sequence-nq'
```

## Usage
```
$ python question_combinator.py --help
usage: question_combinator.py [-h] [--no-except] [--lim-questions LIM_QUESTIONS] [--device {cpu,cuda}]
                              [--models model [model ...]] [--offline] [--rand] [--per-model]
                              [--output-dir OUTPUT_DIR]
                              base_questions_file things_file

Combines questions and data and optionally provides parametric data

positional arguments:
  base_questions_file   File with questions
  things_file           File with things to combine

options:
  -h, --help            show this help message and exit
  --no-except           Do not go to IPDB console on exception.
  --lim-questions LIM_QUESTIONS
                        Question limit
  --device {cpu,cuda}   Inference device
  --models model [model ...]
                        Which model or models to use for getting parametric data
  --offline             Tell HF to run everything offline.
  --rand                Seed randomly
  --per-model           Write one CSV per model in stdout.
  --output-dir OUTPUT_DIR
                        Return one CSV per model, and save them to this directory.
```

## Example usage
```
$ python question_combinator.py --no-except --device cuda --models llama flan-t5-xl flan-t5-xxl --output-dir outputs -- data/base_questions.txt data/objects.csv
```

## Current list of models
```
$ python Models.py --list-models
     Model Name | Huggingface Model
----------------|-----------------------------------------
          llama | meta-llama/Meta-Llama-3.1-8B-Instruct
      llama-70b | meta-llama/Meta-Llama-3.1-70B-Instruct
     llama-405b | meta-llama/Meta-Llama-3.1-405B-Instruct
        flan-t5 | google/flan-t5-base
  flan-t5-small | google/flan-t5-small
   flan-t5-base | google/flan-t5-base
  flan-t5-large | google/flan-t5-large
     flan-t5-xl | google/flan-t5-xl
    flan-t5-xxl | google/flan-t5-xxl
          gemma | google/gemma-2-9b-it
      gemma-27b | google/gemma-2-27b-it
        falcon2 | tiiuae/falcon-11b
    falcon-180b | tiiuae/falcon-180b-chat
     falcon-40b | tiiuae/falcon-40b-instruct
      falcon-7b | tiiuae/falcon-7b-instruct
     distilbert | distilbert/distilbert-base-uncased-distilled-squad
        roberta | FacebookAI/roberta-base
  roberta-large | FacebookAI/roberta-large
  roberta-squad | deepset/roberta-base-squad2
        mixtral | mistralai/Mixtral-8x22B-Instruct-v0.1
          dummy |
```
