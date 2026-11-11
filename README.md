# RAG FAQ bot

Ассистент по документам: загружаешь PDF/TXT — задаёшь вопросы, получаешь ответы
со ссылками на источники. Классический RAG-пайплайн без тяжёлых фреймворков.

## Как работает

1. **Ingest** (один раз): документ → нарезка на чанки (800 символов, overlap 150) →
   эмбеддинги → векторная БД Chroma.
2. **Query** (на каждый вопрос): вопрос → эмбеддинг → поиск top-4 ближайших чанков →
   промпт «контекст + вопрос» → LLM → ответ + источники.

## Стек

- **OpenRouter** (OpenAI-совместимый API): chat `nvidia/nemotron-3.5-super-vl:free`,
  embeddings `nvidia/nemotron-3.5-embed-1b:free`
- **Chroma** — локальная векторная БД (`chroma_db/`)
- **FastAPI** — REST API, **langchain-text-splitters** — нарезка, **pypdf** — чтение PDF

## Установка

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env           # и вставить свой OPENROUTER_API_KEY
```

## Использование

CLI:

```bash
python main.py ingest docs/test.txt   # залить документ
python main.py                        # задать вопрос интерактивно
python main.py reset                  # очистить базу
```

REST API:

```bash
uvicorn api:app --reload
# Swagger: http://127.0.0.1:8000/docs
```

Эндпоинты:

| Метод  | Путь               | Что делает                                  |
|--------|--------------------|---------------------------------------------|
| POST   | `/upload`          | загрузить PDF/TXT и залить в БД             |
| POST   | `/ask`             | ответить на вопрос по документам            |
| GET    | `/chunks/{index}`  | посмотреть чанк по индексу (для отладки)    |
| POST   | `/reset`           | очистить векторную БД                       |
| GET    | `/health`          | проверка живости                            |
| GET    | `/v1/models`       | список моделей (OpenAI-совместимо)          |

## Структура

```
rag/
├── config.py   # клиент OpenRouter, модели, сплиттер, Chroma
├── ingest.py   # загрузка документов → чанки → векторы → БД
└── query.py    # retrieval → промпт → LLM → ответ с источниками
main.py         # CLI
api.py          # FastAPI
```
