# -*- coding: utf-8 -*-
"""Общие фикстуры pytest.

Главная идея: тесты не должны ходить в сеть и не должны трогать реальную
базу в ./chroma_db. Поэтому:
  - Chroma подменяется на EphemeralClient (в памяти, умирает с тестом);
  - вызовы эмбеддингов подменяются заглушкой.
"""
import os
import uuid

import chromadb
import pytest

# rag.config требует ключ уже на импорте — подставляем фейковый ДО импорта
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

from rag import config  # noqa: E402


@pytest.fixture
def temp_collection(monkeypatch):
    """Изолированная коллекция в памяти на время одного теста."""
    client = chromadb.EphemeralClient()
    coll = client.get_or_create_collection(name=f"test_{uuid.uuid4().hex}")
    monkeypatch.setattr(config, "collection", coll)
    return coll


@pytest.fixture
def fake_embed(monkeypatch):
    """Заглушка эмбеддингов: одинаковый вектор на любой текст.

    Размерность 2048 — как у nvidia/nemotron-3.5-embed-1b, чтобы Chroma
    не ругалась на несовпадение размерностей.
    """
    def _embed(texts):
        return [[0.01] * 2048 for _ in texts]

    monkeypatch.setattr("rag.ingest.embed", _embed)
    return _embed
