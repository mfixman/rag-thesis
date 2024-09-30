import csv
import logging
import itertools
import random
import time
import typing

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Any

# Custom filter that does not print a log if it printed another one at most `rate_limit` seconds ago.
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

# A question contains combines a base_question and an object into something that can be queried.
@dataclass
class Question:
    category: str
    obj: str
    base_question: str

    # Static constructor: return a question combining an object and an object if the category
    # matches; return None otherwise.
    @staticmethod
    def orNothing(obj: str, category: str, base_question: str) -> Optional['Question']:
        if not f'{{{category}}}' in base_question:
            return None

        return Question(obj = obj, category = category, base_question = base_question)

    # Return a query from the format of this Question.
    def format(self, *, prompt: Optional[str] = None, context: Optional[str] = None, use_question: bool = True, use_later: bool = True) -> str:
        [question, later] = self.base_question.format_map({self.category: self.obj}).split('?', 1)
        question += '?'

        formatted = ''
        if use_question:
            formatted = f'Q: {question.strip()}'

        if use_later:
            formatted = f'{formatted} A: {later.strip()}'

        if prompt is not None:
            formatted = f'{prompt} {formatted}'

        if context is not None:
            formatted = f'Context: [{later.strip()} {context}]. {formatted}'

        return formatted.strip()

# Returns the set product of a list of base question with the respective set of objects.
def combine_questions(base_questions: list[str], objects: list[dict[str, str]], lim_questions: Optional[int] = None) -> list[Question]:
    questions = []
    for bq in base_questions:
        for obj in objects:
            q = Question.orNothing(obj = obj['object'], category = obj['category'], base_question = bq)
            if q is None:
                continue

            questions.append(q)

    if lim_questions is None:
        return questions

    keep_nums = {x: e for e, x in enumerate(random.sample(range(len(questions)), lim_questions))}
    short_questions = [questions[x] for x in keep_nums.keys()]

    return short_questions

# Given a list of questions and a list of answers, produce a list of integers that would provide the
# index to a randomly sampled counterfactual.
def sample_counterfactual_flips(questions: list[Question], answers: list[str]) -> list[int]:
    flips = [-1 for _ in questions]

    for q, es_iter in itertools.groupby(range(len(questions)), key = lambda e: questions[e].base_question):
        es = set(es_iter)

        for e in es:
            rest = [x for x in es if answers[x] != answers[e]]
            if not rest:
                logging.error(f'Unitary question "{q}". This means that all answers in this chunk are identical, and the results will be incorrect.')
                flips[e] = e
                continue

            flips[e] = random.choice(rest)
            assert answers[flips[e]] != answers[e]

    assert all(x != -1 for x in flips)
    return flips

# Chunk a list of question into batches of size or at most `max_batch_size`.
def chunk_questions(questions: list[Question], max_batch_size: int) -> list[list[Question]]:
    result: list[list[Question]] = []

    for q, chunk_iter in itertools.groupby(questions, key = lambda x: x.base_question):
        chunk = list(chunk_iter)
        if not result or len(chunk) + len(result[-1]) > max_batch_size:
            result.append([])

        result[-1].extend(chunk)

    return result

# Prints a CSV file with the questions and resulting answers.
def print_parametric_csv(out: typing.TextIO, answer: dict[str, list[Any]]):
    fieldnames = ['Num', 'Category', 'Base_Question', 'Object', 'Question', 'Prefix'] + list(answer.keys())

    writer = csv.DictWriter(
        out,
        fieldnames = fieldnames,
        extrasaction = 'ignore',
        dialect = csv.unix_dialect,
        quoting = csv.QUOTE_MINIMAL,
    )
    writer.writeheader()

    for e, answers in enumerate(zip(*answer.values())):
        param = dict(zip(answer.keys(), answers))

        question = param['question']
        param.pop('question')

        writer.writerow(
            {
                'Num': str(e),
                'Category': question.category,
                'Base_Question': ''.join(question.base_question.partition('?')[0:2]),
                'Object': question.obj,
                'Question': question.format(use_later = False),
                'Prefix': question.format(use_question = False)
            } | param
        )
