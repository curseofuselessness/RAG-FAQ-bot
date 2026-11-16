"""REST API + веб-интерфейс: uvicorn api:app --reload → http://127.0.0.1:8000"""
import json
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag.ingest import ingest_file, list_documents
from rag import config
from rag.config import CHAT_MODEL, EMBED_MODEL
from rag.query import answer, retrieve, stream_answer

BASE_DIR = Path(__file__).parent

app = FastAPI(title="RAG Assistant")

# Раздаём папку static целиком: /static/app.js и т.п.
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class AskRequest(BaseModel):
    """Тело запроса к /ask: просто текст вопроса."""

    question: str


class AskResponse(BaseModel):
    """Формат ответа /ask: текст ответа, источники и расход токенов."""

    answer: str
    sources: list[dict]
    usage: dict


@app.get("/", include_in_schema=False)
async def index():
    """Отдаёт веб-интерфейс (static/index.html) на главной странице."""
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/upload")
async def upload(file: UploadFile):
    """Принимает файл (PDF/TXT), сохраняет в uploads/ и заливает в Chroma.

    Возвращает имя файла и число созданных чанков.
    """
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(400, "Only PDF/TXT")
    os.makedirs("uploads", exist_ok=True)
    # basename защищает от имени вида "..\..\secret.txt" — запишем только имя
    safe_name = os.path.basename(file.filename)
    path = os.path.join("uploads", safe_name)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chunks = ingest_file(path)
    return {"filename": safe_name, "chunks": chunks}


@app.get("/documents")
async def documents():
    """Список загруженных документов и число чанков у каждого — для UI и отладки."""
    return {"documents": list_documents()}


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


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """То же, что /ask, но отдаёт ответ потоком (Server-Sent Events).

    Формат каждого события: `data: {"type": ..., ...}\\n\\n`
      type=sources — найденные источники (приходят первыми, до генерации)
      type=token   — очередной кусок текста ответа
      type=done    — конец ответа
      type=error   — текст ошибки
    """
    if not req.question.strip():
        raise HTTPException(400, "Empty question")

    def event(payload: dict) -> str:
        # SSE: строка "data: <json>" + пустая строка-разделитель
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def generate():
        try:
            # retrieval уже выполнен — источники можно отдать сразу
            sources = [{"source": c["source"], "chunk": c["chunk"]} for c in retrieve(req.question)]
            yield event({"type": "sources", "sources": sources})
            for token in stream_answer(req.question):
                yield event({"type": "token", "text": token})
            yield event({"type": "done"})
        except Exception as exc:  # сеть/лимиты OpenRouter — не роняем поток
            yield event({"type": "error", "message": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health():
    """Проверка живости сервера для мониторинга/балансировщиков."""
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    """OpenAI-совместимый список моделей"""
    return {"object": "list", "data": [{"id": m, "object": "model"} for m in (CHAT_MODEL, EMBED_MODEL)]}
