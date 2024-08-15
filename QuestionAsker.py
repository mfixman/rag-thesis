from typing import IO, Any, Union
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

    def prepareQuestions(self, prompt: str, questions: list[str]) -> dict[tuple[str, str], str]:
        enhanced_questions: dict[tuple[str, str], str] = {}
        for q in questions:
            for rag in self.rags:
                context = rag.retrieve_context(q)
                enhanced_questions[(q, rag.name())] = prompt.format(context = context, question = q)

        return enhanced_questions

    # def findAnswers(self, answerer: QuestionAnswerer, questions: list[str], custom_prompt: str):
    #     enhanced_questions = self.getQuestions(custom_prompt, questions)
    #     answers: dict[str, dict[str, str | float]] = answerer.query(enhanced_questions)

    #     for question, row in answers.items():
    #         self.writer.writerow(
    #             {'Question': question}
    #             | {prev.name: prev.retrieve_context(question) for prev in prevs}
    #             | row
    #         )
