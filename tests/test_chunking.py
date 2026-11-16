# -*- coding: utf-8 -*-
"""Чанкинг: размеры кусков и сохранность текста."""
from rag.config import splitter


def test_chunks_do_not_exceed_limit():
    text = "Это предложение. " * 300  # ~5400 символов
    chunks = splitter.split_text(text)

    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)


def test_text_is_not_lost():
    text = "Первое предложение. Второе предложение. Третье предложение."
    chunks = splitter.split_text(text)

    # весь исходный текст встречается в склейке (пробелы могут отличаться)
    joined = "".join(chunks).replace(" ", "")
    assert joined == text.replace(" ", "")


def test_short_text_stays_single_chunk():
    text = "НДФЛ удерживается по ставке 13 процентов."
    assert splitter.split_text(text) == [text]


def test_empty_text_gives_no_chunks():
    assert splitter.split_text("") == []
