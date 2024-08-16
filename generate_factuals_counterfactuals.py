from argparse import ArgumentParser, RawDescriptionHelpFormatter
from csv import DictReader
from dataclasses import dataclass
from datetime import datetime

import random
import sys

@dataclass
class DataPoint:
    columnName: str
    question: str
    statement: str
    is_date: bool = False

    def show(self):
        return f'{self.columnName:<17} | {self.question:>40} | {self.statement:>41}'

DataPoints = [
    DataPoint('dateOfBirth', 'What is the date of birth of {person}?', 'The year of birth of {person} is {data}', is_date = True),
    DataPoint('cityOfBirthLabel', 'In what city was {person} born?', '{person} was born in {data}'),
    DataPoint('dateOfDeath', 'What is the date of death of {person}?', 'The date of death of {person} was on {data}', is_date = True),
    DataPoint('causeOfDeathLabel', 'What is the cause of death of {person}?', 'The cause of death of {person} was by {data}'),
]

def parse_args():
    parser = ArgumentParser(
        description = 'Generates a list of factuals and counterfactuals from a CSV data file.',
        formatter_class = RawDescriptionHelpFormatter,
        epilog = f'Current questions:\n' + '\n'.join(q.show() for q in DataPoints)
    )
    parser.add_argument('input_file', type = open, help = 'CSV file with input data.')

    return parser.parse_args()

def main():
    args = parse_args()
    data = list(DictReader(args.input_file, dialect = 'unix'))

    questions_file = open('datas/questions.txt', 'w')
    factuals_file = open('datas/factuals.txt', 'w')
    counterfactuals_file = open('datas/counterfactuals.txt', 'w')
    for e, row in enumerate(data):
        person = row['personLabel']
        for point in DataPoints:
            factual = row[point.columnName]
            counterfactual = data[random.choice(list(set(range(len(data))) - {e}))][point.columnName]

            if point.is_date:
                factual = datetime.fromisoformat(factual).strftime('%d %B %Y')
                counterfactual = datetime.fromisoformat(counterfactual).strftime('%d %B %Y')

            full_question = f'{point.question} {point.statement}'
            print(full_question.format(person = person, data = ''), file = questions_file)
            print(point.statement.format(person = person, data = factual), file = factuals_file)
            print(point.statement.format(person = person, data = counterfactual), file = counterfactuals_file)

    print(f'Succesfully wrote {questions_file.name}, {factuals_file.name}, and {counterfactuals_file.name}.', file = sys.stderr)

if __name__ == '__main__':
    main()
