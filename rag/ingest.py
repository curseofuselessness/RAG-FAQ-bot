"""Загрузка документов, эмбеддинги и запись в Chroma (режим ingest)."""
from pypdf import PdfReader

from rag import config
from rag.config import client, splitter, EMBED_MODEL


def load_pdf(path: str) -> str:
    """Читает PDF и возвращает весь текст одной строкой (страницы склеены через \\n)."""
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_text(path: str) -> str:
    """Читает .txt файл целиком в кодировке UTF-8 и возвращает его содержимое."""
    with open(path, encoding="utf-8") as f:
        return f.read()


def embed(texts: list[str]) -> list[list[float]]:
    """Превращает список текстов в список эмбеддингов через OpenRouter.

    API принимает input как список (можно много текстов за один запрос —
    так дешевле и быстрее при ingest) и возвращает по одному вектору на каждый.
    """
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def ingest(chunks: list[str], source_name: str) -> int:
    """Кладёт чанки в Chroma, заменяя предыдущую версию этого источника.

    Идемпотентность: повторная загрузка файла с тем же именем не плодит дубли
    и не падает на Duplicate ID — старые чанки удаляются, новые добавляются.

    Порядок операций важен: эмбеддинги (сеть, самое ненадёжное) считаем ДО
    удаления, чтобы при сбое OpenRouter старые данные остались на месте.

    Возвращает число добавленных чанков.
    """
    if not chunks:
        raise ValueError("В файле не найден текст (возможно, это скан без текстового слоя)")

    embeddings = embed(chunks)                              # 1. сеть: может упасть — база цела
    config.collection.delete(where={"source": source_name})  # 2. убрать старую версию файла
    config.collection.add(                                  # 3. записать новую
        ids=[f"{source_name}_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,  # сам текст — нужен для промпта при retrieval
        metadatas=[{"source": source_name, "chunk": i} for i in range(len(chunks))],
    )
    return len(chunks)


def ingest_file(path: str) -> int:
    """Полный пайплайн для одного файла: прочитать → нарезать на чанки → залить в Chroma."""
    text = load_pdf(path) if path.endswith(".pdf") else load_text(path)
    chunks = splitter.split_text(text)
    return ingest(chunks, source_name=path.replace("\\", "/").split("/")[-1])


def list_documents() -> list[dict]:
    """Возвращает источники, лежащие в БД, и число чанков у каждого.

    Chroma умеет отдать только метаданные (без текстов и векторов) —
    этого достаточно, чтобы посчитать чанки по каждому файлу.
    """
    res = config.collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in res["metadatas"]:
        source = meta.get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return [{"source": s, "chunks": n} for s, n in sorted(counts.items())]
