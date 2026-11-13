"""REST API: uvicorn api:app --reload → http://127.0.0.1:8000/docs"""
import os
import shutil

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from rag.ingest import ingest_file
from rag import config
from rag.config import CHAT_MODEL, EMBED_MODEL
from rag.query import answer

app = FastAPI(title="RAG Assistant")


class AskRequest(BaseModel):
    """Тело запроса к /ask: просто текст вопроса."""

    question: str


class AskResponse(BaseModel):
    """Формат ответа /ask: текст ответа, источники и расход токенов."""

    answer: str
    sources: list[dict]
    usage: dict


@app.post("/upload")
async def upload(file: UploadFile):
    """Принимает файл (PDF/TXT), сохраняет в uploads/ и заливает в Chroma.

    Возвращает имя файла и число созданных чанков.
    """
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(400, "Only PDF/TXT")
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chunks = ingest_file(path)
    return {"filename": file.filename, "chunks": chunks}


@app.get("/chunks/{index}")
async def get_chunk(index: int):
    """Возвращает чанк по его индексу в коллекции (0, 1, 2, ...).

    Полезно для отладки retrieval: посмотреть, что реально лежит в БД.
    404, если индекс за пределами коллекции.
    """
    total = config.collection.count()
    if index < 0 or index >= total:
        raise HTTPException(404, f"Index {index} out of range, collection size = {total}")
    # Chroma умеет брать срез: limit=1, offset=index → ровно один чанк
    res = config.collection.get(
        limit=1,
        offset=index,
        include=["documents", "metadatas"],
    )
    return {
        "index": index,
        "id": res["ids"][0],
        "metadata": res["metadatas"][0],
        "text": res["documents"][0],
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Отвечает на вопрос по загруженным документам (полный RAG-цикл)."""
    if not req.question.strip():
        raise HTTPException(400, "Empty question")
    return answer(req.question)

@app.get("/health")
async def health():
    """Проверка живости сервера для мониторинга/балансировщиков."""
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    """OpenAI-совместимый список моделей"""
    return {"object": "list", "data": [{"id": m, "object": "model"} for m in (CHAT_MODEL, EMBED_MODEL)]}

