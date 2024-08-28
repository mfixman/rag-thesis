from RagEnhancer import RAG

import csv
import logging
import itertools
import random
import typing
import sys

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Any

from torch import FloatTensor, LongTensor, BoolTensor

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
            formatted = f'Context: [{later.strip()} {context}]. {formatted}'

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
            cat_positions[obj.question].add(len(questions) - 1)

    if lim_questions is None:
        return questions, cat_positions

    keep_nums = {x: e for e, x in enumerate(random.sample(range(len(questions)), lim_questions))}
    short_questions = [questions[x] for x in keep_nums.keys()]
    short_cat_positions = {k: {keep_nums[t] for t in v & set(keep_nums)} for k, v in cat_positions.items() if not v.isdisjoint(keep_nums)}

    return short_questions, short_cat_positions

def find_flips(cat_positions: dict[str, set[int]], total: int) -> list[int]:
    flips: list[Optional[int]] = [None for _ in range(total)]
    for cat, values in cat_positions.items():
        assert len(values) > 0

        if len(values) == 1:
            logging.warn('Question without counterfactual. Keeping it equal!')
            v = next(iter(values))
            flips[v] = v
            continue

        for v in values:
            flips[v] = random.choice(list(values - {v}))

    assert all(x is not None for x in flips)
    return typing.cast(list[int], flips)

def findFlips2(questions: list[Object]) -> list[int]:
    flips = [-1 for _ in questions]

    for cat, es_iter in itertools.groupby(range(len(questions)), key = lambda e: questions[e].category):
        es = set(es_iter)

        if len(es) == 1:
            logging.warn(f'Unitary category {cat}. id flip!')
            e = next(iter(es))
            flips[e] = e
            continue

        for e in es:
            flips[e] = random.choice(list(es - {e}))

    assert all(x != -1 for x in flips)
    return flips

def printParametricCSV(out: typing.TextIO, questions: list[Object], answer: dict[str, str]):
    fieldnames = ['Num', 'Category', 'Base_Question', 'Thing', 'Question', 'Prefix'] + list(answer.keys())

    writer = csv.DictWriter(
        out,
        fieldnames = fieldnames,
        extrasaction = 'ignore',
        dialect = csv.unix_dialect,
        quoting = csv.QUOTE_MINIMAL,
    )
    writer.writeheader()

    for e, (question, *answers) in enumerate(itertools.zip_longest(questions, *answer.values())):
        question = typing.cast(Object, question)

        param = dict(zip(answer.keys(), answers))
        writer.writerow({'Num': str(e), 'Category': question.category, 'Base_Question': ''.join(question.question.partition('?')[0:2]), 'Thing': question.thing, 'Question': question.format(use_later = False), 'Prefix': question.format(use_question = False)} | param)
