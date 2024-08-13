import csv
from typing import IO

def getWriter(models: list[str], include_logits: bool, raw_answer_files: None | list[IO[str]]) -> csv.DictWriter:
    fieldnames = [f'{n}_{llm}' for llm in models for n in rag.names() for rag in rags]
    if include_logits:
        fieldnames = list(itertools.chain(*[[f'{x}', f'{x}_logits'] for x in fieldnames]))

    if raw_answers_files is not None:
        fieldnames = [Path(x.name).stem for x in raw_answers_files] + fieldnames

    return csv.DictWriter(
        sys.stdout,
        fieldnames = ['Question'] + fieldnames,
        extrasaction = 'ignore',
        dialect = csv.unix_dialect,
        quoting = csv.QUOTE_MINIMAL,
    )

def askQuestions(
    answerer: 'QuestionAnswerer',
    questions: list[str],
    rags: list['RAG'],
    answer_files: list[IO[str]],
    custom_prompt: str
    ) -> list[dict[str, any]]:
    answer_files = args.raw_answers_files or []
    filenames = [Path(x.name).stem for x in answer_files]

    rows = []
    for question, *raw_answers in zip(questions, *answer_files):
        raw_answers = [re.sub(r'.*(was by|was on|is|in) +', '', a.strip()) for a in raw_answers]

        results = {'Question': question} | dict(zip(filenames, raw_answers))
        for rag in rags:
            context = rag.retrieve_context(question)
            enhanced_question = custom_prompt.format(context = context, question = question)
            answers, rest = answerer.query(enhanced_question, short = True, return_rest = True)

            for llm, answer in answers.items():
                name = f'{rag.name()}_{llm}'
                results[name] = answer

                if args.logits:
                    results[f'{name}_logits'] = round(rest[llm]['logit_prod'], 2)

        rows.append({'Question': question} | results)

    return rows
