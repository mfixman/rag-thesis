from RagEnhancer import RAG

import csv
import logging
import itertools
import sys

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
