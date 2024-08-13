from typing import IO, Any
from RagEnhancer import RAG
from QuestionAnswerer import QuestionAnswerer

from pathlib import Path
import csv
import logging
import itertools
import sys

from typing import Optional

class QuestionAsker:
    def __init__(self, model_names: list[str], prevs: list[RAG], rags: list[RAG], include_logits: bool):
        self.prevs = prevs
        self.rags = rags

        fieldnames = [f'{rag.name()}-{llm}' for llm in model_names for rag in self.rags]
        if include_logits:
            fieldnames = list(itertools.chain(*[[f'{x}', f'{x}-logits'] for x in fieldnames]))

        fieldnames = ['Question'] + [f'{prev.name()}' for prev in prevs] + fieldnames

        self.writer = csv.DictWriter(
            sys.stdout,
            fieldnames = fieldnames,
            extrasaction = 'ignore',
            dialect = csv.unix_dialect,
            quoting = csv.QUOTE_MINIMAL,
        )
        self.writer.writeheader()

    def findAnswers(self, answerer: QuestionAnswerer, questions: list[str], custom_prompt: str):
        for e, question in enumerate(questions, start = 1):
            if e % 10 == 0:
                logging.info(f'Question {e}/{len(questions)}')

            results = {'Question': question}
            for prev in self.prevs:
                results[prev.name()] = prev.retrieve_context(question)

            for rag in self.rags:
                context = rag.retrieve_context(question)
                enhanced_question = custom_prompt.format(context = context, question = question)
                answers, rest = answerer.query(enhanced_question)

                for llm, answer in answers.items():
                    name = f'{rag.name()}-{llm}'
                    results[name] = answer
                    results[f'{name}_logits'] = round(rest[llm]['logit_prod'], 2)

            self.writer.writerow({'Question': question} | results)
