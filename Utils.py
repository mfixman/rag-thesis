from RagEnhancer import RAG

import csv
import logging
import itertools
import random
import typing
import sys

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

def prepareQuestions(rags: list[RAG], prompt: str, questions: list[str]) -> dict[tuple[str, str], str]:
    enhanced_questions: dict[tuple[str, str], str] = {}
    for q in questions:
        for rag in rags:
            context = rag.retrieve_context(q)
            enhanced_questions[(q, rag.name())] = prompt.format(context = context, question = q)

    return enhanced_questions

def printCSV(
    input_files: list[RAG],
    data: dict[tuple[str, str], tuple[str, float]],
    include_logits: bool,
):
    initial_row = next(iter(data))[0]
    data_columns = [col for row, col in data.keys() if row == initial_row]
    if include_logits:
        data_columns = list(itertools.chain.from_iterable(zip(data_columns, [x + '-logit' for x in data_columns])))

    fieldnames = ['Question'] + [data.name() for data in input_files] + data_columns
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames = fieldnames,
        dialect = csv.unix_dialect,
        quoting = csv.QUOTE_MINIMAL,
        extrasaction = 'ignore',
    )
    writer.writeheader()

    for row, cols in itertools.groupby(data.items(), key = lambda x: x[0][0]):
        values = [[(col, answer), (col + '-logit', round(logit, 3))] for ((_, col), (answer, logit)) in cols]
        writer.writerow(
            {'Question': row} |
            {line.name(): line.retrieve_context(row) for line in input_files} |
            dict(itertools.chain.from_iterable(values))
        )

@dataclass
class Object:
    thing: str
    category: str
    question: str

    @staticmethod
    def orNothing(thing: str, category: str, question: str) -> Optional['Object']:
        if not f'{{{category}}}' in question:
            return None

        return Object(thing = thing, category = category, question = question)

    def format(self, *, prompt: Optional[str] = None, context: Optional[str] = None, use_question: bool = True, use_later: bool = True) -> str:
        [question, later] = self.question.format_map({self.category: self.thing}).split('?', 1)
        question += '?'

        formatted = ''
        if use_question:
            formatted = f'{question.strip()}'

        if use_later:
            formatted = f'{formatted} {later.strip()}'

        if prompt is not None:
            formatted = f'{prompt} {formatted}'

        if context is not None:
            formatted = f'Context: [{context}]. {formatted}'

        return formatted.strip()

def combine_questions(base_questions: list[str], things: list[dict[str, str]], lim_questions: Optional[int] = None) -> tuple[list[Object], dict[str, set[int]]]:
    questions = []
    cat_positions = defaultdict(lambda: set())
    for bq in base_questions:
        for thing in things:
            obj = Object.orNothing(thing = thing['thing'], category = thing['category'], question = bq)
            if obj is None:
                continue

            questions.append(obj)
            cat_positions[obj.category].add(len(questions) - 1)

            if len(questions) == lim_questions:
                return questions, cat_positions

    return questions, cat_positions

def find_flips(cat_positions: dict[str, set[int]], total: int) -> list[int]:
    flips: list[Optional[int]] = [None for _ in range(total)]
    for cat, values in cat_positions.items():
        for v in values:
            flips[v] = random.choice(list(values - {v}))

    assert all(x is not None for x in flips)
    return typing.cast(list[int], flips)

def printParametricCSV(questions: list[Object], parametric: dict[str, str], counterfactuals: dict[str, str] = {}):
    fieldnames = ['Category', 'Question', 'Prefix'] + list(parametric.keys())
    if counterfactuals:
        fieldnames += list(itertools.chain.from_iterable([[c, 'cf_context-' + c] for c in counterfactuals.keys()]))

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames = fieldnames,
        extrasaction = 'ignore',
        dialect = csv.unix_dialect,
        quoting = csv.QUOTE_MINIMAL,
    )
    writer.writeheader()

    for question, *answers in itertools.zip_longest(questions, *parametric.values(), *counterfactuals.values()):
        question = typing.cast(Object, question)

        param = dict(zip(parametric.keys(), answers))
        counter = dict(zip(counterfactuals.keys(), answers[len(parametric):]))
        writer.writerow({'Category': question.category, 'Question': question.format(use_later = False), 'Prefix': question.format(use_question = False)} | param | counter)
