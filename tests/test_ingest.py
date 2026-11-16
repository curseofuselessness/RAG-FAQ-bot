# -*- coding: utf-8 -*-
"""Ingest: идемпотентность, метаданные, отказ на пустом документе."""
import pytest

from rag.ingest import ingest


def test_ingest_adds_chunks(temp_collection, fake_embed):
    n = ingest(["первый текст", "второй текст"], source_name="a.txt")

    assert n == 2
    assert temp_collection.count() == 2


def test_ingest_is_idempotent(temp_collection, fake_embed):
    ingest(["старый текст"], source_name="a.txt")
    ingest(["новый текст"], source_name="a.txt")

    # запись одна и она новая — не две версии сразу
    assert temp_collection.count() == 1
    stored = temp_collection.get(where={"source": "a.txt"}, include=["documents"])
    assert stored["documents"] == ["новый текст"]


def test_ingest_replaces_longer_document(temp_collection, fake_embed):
    """Старый документ был из 3 чанков, новый из 1 — лишние чанки не должны остаться."""
    ingest(["a", "b", "c"], source_name="a.txt")
    ingest(["только один"], source_name="a.txt")

    assert temp_collection.count() == 1


def test_ingest_keeps_other_sources(temp_collection, fake_embed):
    ingest(["документ A"], source_name="a.txt")
    ingest(["документ B"], source_name="b.txt")
    ingest(["документ A, версия 2"], source_name="a.txt")

    sources = temp_collection.get(include=["metadatas"])["metadatas"]
    names = sorted(m["source"] for m in sources)

    assert names == ["a.txt", "b.txt"]


def test_ingest_sets_metadata(temp_collection, fake_embed):
    ingest(["один", "два"], source_name="doc.txt")

    metas = sorted(
        temp_collection.get(include=["metadatas"])["metadatas"],
        key=lambda m: m["chunk"],
    )
    assert metas == [{"source": "doc.txt", "chunk": 0}, {"source": "doc.txt", "chunk": 1}]


def test_ingest_rejects_empty_document(temp_collection, fake_embed):
    with pytest.raises(ValueError, match="не найден текст"):
        ingest([], source_name="empty.txt")

    assert temp_collection.count() == 0
