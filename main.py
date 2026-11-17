"""CLI: `python main.py ingest <файл>` — залить документ, без аргументов — режим вопросов."""
import sys

from rag.ingest import ingest_file
from rag.config import MAX_DISTANCE, reset_db
from rag.query import answer, retrieve

if __name__ == "__main__":
    # Два режима:
    #   ingest — одноразовый: документ → чанки → векторы → Chroma
    #   без аргументов — на каждый вопрос: retrieve top-k → промпт → LLM
    if len(sys.argv) > 2 and sys.argv[1] == "ingest":
        try:
            n = ingest_file(sys.argv[2])
            print(f"Ingested {n} chunks from {sys.argv[2]}")
        except ValueError as exc:
            print(f"Ошибка: {exc}")
            sys.exit(1)
    elif len(sys.argv) > 1 and sys.argv[1] == "reset":
        from rag import config as cfg

        print(f"Deleted {cfg.collection.count()} chunks")
        reset_db()
        print("БД очищена")
    elif len(sys.argv) > 2 and sys.argv[1] == "debug":
        # Показать близость всех top-k чанков без фильтра по порогу
        question = sys.argv[2]
        chunks = retrieve(question, apply_threshold=False)
        print(f"\nВопрос: {question}")
        print(f"MAX_DISTANCE = {MAX_DISTANCE}\n")
        for i, c in enumerate(chunks, 1):
            mark = "ok " if c["distance"] <= MAX_DISTANCE else "far"
            print(f"{i}. [{mark}] distance={c['distance']:.4f}  {c['source']} chunk {c['chunk']}")
            print(f"   {c['text'][:200].replace(chr(10), ' ')}…\n")
    else:
        question = input("Вопрос: ")
        result = answer(question)
        print("\nОтвет:", result["answer"])
        print("\nИсточники:", result["sources"])
        print("Токены:", result["usage"])
