# -*- coding: utf-8 -*-
"""Калибровка порога релевантности (MAX_DISTANCE).

Печатает distance лучшего чанка для каждого вопроса. Порог ставится
в «зазоре» между числами вопросов, у которых ответ в документах есть,
и вопросов, у которых ответа нет.

Запуск:
    python eval/calibrate.py                  # вопросы из eval/questions.txt
    python eval/calibrate.py my_questions.txt  # свой список

Одна строка = один вопрос. Строки, начинающиеся с #, игнорируются.
"""
import sys

from rag.query import retrieve


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "eval/questions.txt"
    with open(path, encoding="utf-8") as f:
        questions = [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    print(f"{'distance':>8}  {'источник':<28}  вопрос")
    print("-" * 78)

    for question in questions:
        # apply_threshold=False: нужны сырые числа, а не отфильтрованные
        chunks = retrieve(question, apply_threshold=False)
        if not chunks:
            print(f"{'—':>8}  {'—':<28}  {question}")
            continue
        best = chunks[0]
        print(f"{best['distance']:>8.3f}  {best['source']:<28}  {question}")


if __name__ == "__main__":
    main()
