from typing import IO, Any
from RagEnhancer import RAG
from QuestionAnswerer import QuestionAnswerer

from pathlib import Path
import csv
import logging
import itertools
import re
import sys

from typing import Optional

class QuestionAsker:
    def __init__(self, models: list[str], rags: list[RAG], answer_files: Optional[list[IO[str]]], include_logits: bool):
        self.rags = rags
        self.answer_files = answer_files or []
        self.include_raw_answers = answer_files is not None

        fieldnames = [f'{n}_{llm}' for llm in models for rag in self.rags for n in rag.names()]
        if include_logits:
            fieldnames = list(itertools.chain(*[[f'{x}', f'{x}_logits'] for x in fieldnames]))

        if self.include_raw_answers:
            fieldnames = [Path(x.name).stem for x in self.answer_files] + fieldnames

        self.writer = csv.DictWriter(
            sys.stdout,
            fieldnames = ['Question'] + fieldnames,
            extrasaction = 'ignore',
            dialect = csv.unix_dialect,
            quoting = csv.QUOTE_MINIMAL,
        )
        self.writer.writeheader()

    def findAnswers(self, answerer: QuestionAnswerer, questions: list[str], custom_prompt: str):
        filenames = [Path(x.name).stem for x in self.answer_files]

        for e, (question, *raw_answers) in enumerate(zip(questions, *self.answer_files), start = 1):
            if e % 10 == 0:
                logging.info(f'Question {e}/{len(questions)}')

            raw_answers = [re.sub(r'.*(was by|was on|is|in) +', '', a.strip()) for a in raw_answers]

            results = {'Question': question} | dict(zip(filenames, raw_answers))
            for rag in self.rags:
                context = rag.retrieve_context(question)
                enhanced_question = custom_prompt.format(context = context, question = question)
                answers, rest = answerer.query(enhanced_question)

                for llm, answer in answers.items():
                    name = f'{rag.name()}_{llm}'
                    results[name] = answer
                    results[f'{name}_logits'] = round(rest[llm]['logit_prod'], 2)

            self.writer.writerow({'Question': question} | results)
