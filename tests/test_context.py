# -*- coding: utf-8 -*-
"""Сборка контекста и структура источников для промпта."""
from rag.query import build_context


def test_build_context_labels_every_chunk():
    chunks = [
        {"text": "первый", "source": "a.txt", "chunk": 0, "distance": 0.1},
        {"text": "второй", "source": "b.pdf", "chunk": 7, "distance": 0.2},
    ]
    context = build_context(chunks)

    assert "[source: a.txt, chunk 0]" in context
    assert "[source: b.pdf, chunk 7]" in context
    assert "первый" in context and "второй" in context
    # чанки разделены, чтобы модель видела границы
    assert "\n\n---\n\n" in context


def test_build_context_empty():
    assert build_context([]) == ""
