# -*- coding: utf-8 -*-
"""Порог релевантности: фильтрация чанков и отказ без вызова LLM."""
import rag.query as query
from rag.config import MAX_DISTANCE
from rag.query import NO_ANSWER, answer, retrieve


class FakeQueryResult:
    """Подделка ответа Chroma: три чанка с заданными distance."""

    def __init__(self, distances):
        self.distances = distances

    def query(self, **kwargs):
        n = len(self.distances)
        return {
            "documents": [[f"текст {i}" for i in range(n)]],
            "metadatas": [[{"source": "doc.txt", "chunk": i} for i in range(n)]],
            "distances": [self.distances],
        }


def _setup(monkeypatch, distances):
    monkeypatch.setattr(query, "config", type("C", (), {
        "collection": FakeQueryResult(distances),
    }))
    monkeypatch.setattr(query, "embed_query", lambda q: [0.0] * 2048)


def test_retrieve_drops_chunks_beyond_threshold(monkeypatch):
    _setup(monkeypatch, [0.1, MAX_DISTANCE - 0.01, MAX_DISTANCE + 0.5])

    chunks = retrieve("вопрос")

    assert len(chunks) == 2
    assert all(c["distance"] <= MAX_DISTANCE for c in chunks)


def test_retrieve_can_return_raw_distances(monkeypatch):
    """Для калибровки нужны все чанки, без отсечения."""
    _setup(monkeypatch, [0.1, 5.0])

    chunks = retrieve("вопрос", apply_threshold=False)

    assert len(chunks) == 2


def test_retrieve_returns_empty_when_everything_is_far(monkeypatch):
    _setup(monkeypatch, [MAX_DISTANCE + 0.1, MAX_DISTANCE + 0.2])

    assert retrieve("вопрос") == []


def test_answer_skips_llm_without_relevant_chunks(monkeypatch):
    monkeypatch.setattr(query, "retrieve", lambda q, top_k=4: [])

    def boom(**kwargs):
        raise AssertionError("LLM не должна вызываться при пустом контексте")

    monkeypatch.setattr(query.client.chat.completions, "create", boom)

    result = answer("как приготовить борщ?")

    assert result == {"answer": NO_ANSWER, "sources": [], "usage": {}}


def test_answer_calls_llm_with_relevant_chunks(monkeypatch):
    chunks = [{"text": "ставка 13%", "source": "doc.txt", "chunk": 0, "distance": 0.1}]
    monkeypatch.setattr(query, "retrieve", lambda q, top_k=4: chunks)

    class FakeResponse:
        class Choice:
            class Message:
                content = "13 процентов"

            message = Message()

        choices = [Choice()]

        class Usage:
            def model_dump(self):
                return {"total_tokens": 10}

        usage = Usage()

    monkeypatch.setattr(
        query.client.chat.completions, "create", lambda **kwargs: FakeResponse()
    )

    result = answer("какая ставка?")

    assert result["answer"] == "13 процентов"
    assert result["sources"] == [{"source": "doc.txt", "chunk": 0}]
