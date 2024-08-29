from RagEnhancer import RAG

import csv
import logging
import itertools
import random
import time
import typing
import sys

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Any

from torch import FloatTensor, LongTensor, BoolTensor

import ipdb

class LogTimeFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.last_log = defaultdict(lambda: 0)

    def filter(self, record):
        if not hasattr(record, 'rate_limit'):
            return True

        current_time = time.time()
        if current_time - self.last_log[record.lineno] >= record.rate_limit:
            self.last_log[record.lineno] = current_time
            return True

        return False

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

def findFlips2(questions: list[Object]) -> list[int]:
    flips = [-1 for _ in questions]

    for q, es_iter in itertools.groupby(range(len(questions)), key = lambda e: questions[e].question):
        es = set(es_iter)

        if len(es) == 1:
            logging.warn(f'Unitary question "{q}". Identity flip!')
            e = next(iter(es))
            flips[e] = e
            continue

        for e in es:
            flips[e] = random.choice(list(es - {e}))

    assert all(x != -1 for x in flips)
    return flips

def chunkByQuestion(questions: list[Object], max_batch_size: int) -> list[list[Object]]:
    result: list[list[Object]] = []

    for q, chunk_iter in itertools.groupby(questions, key = lambda x: x.question):
        chunk = list(chunk_iter)
        if not result or len(chunk) + len(result[-1]) > max_batch_size:
            result.append([])

        result[-1].extend(chunk)

    return result


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
