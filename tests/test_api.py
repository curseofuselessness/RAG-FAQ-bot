# -*- coding: utf-8 -*-
"""HTTP-слой: статусы, формат ответов, SSE-стриминг."""
import json

import pytest
from fastapi.testclient import TestClient

import api
from rag.ingest import ingest


@pytest.fixture
def client(temp_collection, fake_embed):
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text


def test_upload_accepts_text_file(client):
    r = client.post("/upload", files={"file": ("note.txt", b"privet")})
    assert r.status_code == 200
    assert r.json() == {"filename": "note.txt", "chunks": 1}


def test_upload_rejects_wrong_extension(client):
    r = client.post("/upload", files={"file": ("bad.exe", b"x")})
    assert r.status_code == 400


def test_upload_reports_empty_document(client):
    r = client.post("/upload", files={"file": ("empty.txt", b"")})
    assert r.status_code == 400
    assert "не найден текст" in r.json()["detail"]


def test_upload_uses_basename_only(client):
    """Имя вида ..\\..\\evil.txt не должно выходить за папку uploads."""
    r = client.post("/upload", files={"file": ("..\\..\\evil.txt", b"text")})
    assert r.status_code == 200
    assert r.json()["filename"] == "evil.txt"


def test_documents_lists_sources(client, temp_collection, fake_embed):
    ingest(["текст"], source_name="doc.txt")

    r = client.get("/documents")
    assert r.json() == {"documents": [{"source": "doc.txt", "chunks": 1}]}


def test_chunk_by_index(client, temp_collection, fake_embed):
    ingest(["единственный чанк"], source_name="doc.txt")

    r = client.get("/chunks/0")
    assert r.status_code == 200
    assert r.json()["text"] == "единственный чанк"
    assert r.json()["metadata"]["source"] == "doc.txt"


def test_chunk_index_out_of_range(client, temp_collection, fake_embed):
    ingest(["единственный чанк"], source_name="doc.txt")

    assert client.get("/chunks/99").status_code == 404


def test_ask_returns_answer(client, monkeypatch):
    monkeypatch.setattr(api, "answer", lambda q: {"answer": "42", "sources": [], "usage": {}})

    r = client.post("/ask", json={"question": "в чём смысл?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "42"


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": "   "}).status_code == 400


def test_stream_sends_sources_tokens_and_done(client, monkeypatch):
    monkeypatch.setattr(api, "retrieve", lambda q: [{"source": "a.txt", "chunk": 3}])
    monkeypatch.setattr(api, "stream_answer", lambda q, chunks: iter(["При", "вет"]))

    with client.stream("POST", "/ask/stream", json={"question": "?"}) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())

    payloads = [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]

    assert payloads[0] == {"type": "sources", "sources": [{"source": "a.txt", "chunk": 3}]}
    assert "".join(p["text"] for p in payloads if p["type"] == "token") == "Привет"
    assert payloads[-1] == {"type": "done"}


def test_stream_reports_errors(client, monkeypatch):
    def boom(question, chunks):
        raise RuntimeError("upstream down")

    monkeypatch.setattr(api, "retrieve", lambda q: [{"source": "a.txt", "chunk": 0}])
    monkeypatch.setattr(api, "stream_answer", boom)

    with client.stream("POST", "/ask/stream", json={"question": "?"}) as r:
        text = "".join(r.iter_text())

    assert '"type": "error"' in text
    assert "upstream down" in text


def test_stream_answers_without_llm_when_nothing_relevant(client, monkeypatch):
    monkeypatch.setattr(api, "retrieve", lambda q: [])

    def boom(question, chunks):
        raise AssertionError("LLM не должна вызываться без релевантного контекста")

    monkeypatch.setattr(api, "stream_answer", boom)

    with client.stream("POST", "/ask/stream", json={"question": "борщ?"}) as r:
        body = "".join(r.iter_text())

    payloads = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    assert payloads[0] == {"type": "sources", "sources": []}
    assert any(p.get("text", "").startswith("В документах нет ответа") for p in payloads)
    assert payloads[-1] == {"type": "done"}
