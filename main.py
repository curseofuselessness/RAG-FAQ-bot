"""CLI: `python main.py ingest <файл>` — залить документ, без аргументов — режим вопросов."""
import sys

from rag.ingest import ingest_file
from rag.config import reset_db
from rag.query import answer

if __name__ == "__main__":
    # Два режима:
    #   ingest — одноразовый: документ → чанки → векторы → Chroma
    #   без аргументов — на каждый вопрос: retrieve top-k → промпт → LLM
    if len(sys.argv) > 2 and sys.argv[1] == "ingest":
        n = ingest_file(sys.argv[2])
        print(f"Ingested {n} chunks from {sys.argv[2]}")
    elif len(sys.argv) > 1 and sys.argv[1] == "reset":
        from rag import config as cfg

        print(f"Deleted {cfg.collection.count()} chunks")
        reset_db()
        print("БД очищена")
    else:
        question = input("Вопрос: ")
        result = answer(question)
        print("\nОтвет:", result["answer"])
        print("\nИсточники:", result["sources"])
        print("Токены:", result["usage"])
