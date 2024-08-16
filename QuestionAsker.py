from typing import IO, Any, Union
from RagEnhancer import RAG
from QuestionAnswerer import QuestionAnswerer

from pathlib import Path
import csv
import logging
import itertools
import sys

from typing import Optional

def printCSV(
    input_files: list[RAG],
    data: dict[tuple[str, str], tuple[str, float]],
    include_logits: bool,
):
    initial_row = next(iter(data))[0]
    data_columns = [col for row, col in data.keys() if row == initial_row]
    if include_logits:
        data_columns = list(itertools.chain.from_iterable(zip(data_columns, [x + '-logit' for x in data_columns])))

    logging.info(data_columns)

    fieldnames = ['Question'] + [data.name() for data in input_files] + data_columns
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames = fieldnames,
        dialect = csv.unix_dialect,
        quoting = csv.QUOTE_MINIMAL,
        extrasaction = 'ignore',
    )
    writer.writeheader()

    for row, cols in itertools.groupby(data, key = lambda x: x[0]):
        writer.writerow(
            {'Question': row} |
            {line.name(): line.retrieve_context(row) for line in input_files} |
            dict(itertools.chain.from_iterable([(col, data[(row, col)][0]), (col + '-logit', data[(row, col)][1])] for _, col in cols))
        )

def prepareQuestions(rags: list[RAG], prompt: str, questions: list[str]) -> dict[tuple[str, str], str]:
    enhanced_questions: dict[tuple[str, str], str] = {}
    for q in questions:
        for rag in rags:
            context = rag.retrieve_context(q)
            enhanced_questions[(q, rag.name())] = prompt.format(context = context, question = q)

    return enhanced_questions

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
