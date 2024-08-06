# Enhancing Knowledge Grounding in Retrieval-Augmented Languages Models
An ongoing MSc thesis project by Martin Fixman, supervised by Tillman Weyde.

## What is this about (for now)?
We are competing the grounding of different types of knowledge of many LLMs on several RAG models containing counterfactual information.

Better description to be completer later!

![Data preparation pipeline](figures/Figure_1.png)

![Model evaluation](figures/Figure_2.png)

## Usage
```
(venv) ~/TP/Thesis/Initial_RAG $ python models.py --help
usage: models.py [-h] [--models model [model ...]] [--list-models] [--device {cpu,cuda}] [-l MAX_LENGTH]
                 [--custom-prompt PROMPT] [--empty | --no-empty | --empty-context | --no-empty-context]
                 [--rag | --no-rag] [--rag-dummy | --no-rag-dummy] [--rag-const CONTEXT]
                 [--rag-const-file FILE_WITH_CONTEXT]
                 [question_file]

Ask me a question

positional arguments:
  question_file         File with questions

options:
  -h, --help            show this help message and exit
  --models model [model ...]
                        Which model or models to use
  --list-models         List all available models
  --device {cpu,cuda}
  -l MAX_LENGTH, --max-length MAX_LENGTH
                        Max length of answer
  --custom-prompt PROMPT
                        Use a custom prompt for the questions instead of the default one. {context} and
                        {question} fill to the context and question, respectively
  --empty, --no-empty, --empty-context, --no-empty-context
                        Whether to use an empty context as base
  --rag, --no-rag       Whether to enhance the answer with RAG
  --rag-dummy, --no-rag-dummy
                        Use dummy dataset for RAG
  --rag-const CONTEXT   Mock this context for RAG rather than using a RAG extractor.
  --rag-const-file FILE_WITH_CONTEXT
                        File with data to inject to RAG extractor.

Use the --list-models option for the full list of supported models.

Default prompt:
` ``
Context: [{context}]; Question: [{question}]. Answer briefly using the previous context and without prompting. Answer:
` ``

Example usage: 

# Test llama and falcon2 on the questions in
# datas/questions.txt using both no context and
# the counterfactuals fount in datas/counterfactuals.txt.
python models.py                               \
    --device cuda                              \
    --models llama falcon2                     \
    --empty-context                            \
    --rag-const-file datas/counterfactuals.txt \
    datas/questions.txt
```
