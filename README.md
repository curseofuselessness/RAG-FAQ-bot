# RAG FAQ bot

[![CI](https://github.com/curseofuselessness/RAG-FAQ-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/curseofuselessness/RAG-FAQ-bot/actions/workflows/ci.yml)

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

REST API и веб-интерфейс:

```bash
uvicorn api:app --reload
# Веб-интерфейс: http://127.0.0.1:8000
# Swagger:       http://127.0.0.1:8000/docs
```

Эндпоинты:

| Метод  | Путь               | Что делает                                  |
|--------|--------------------|---------------------------------------------|
| GET    | `/`                | веб-интерфейс (HTML + Tailwind CDN)         |
| POST   | `/upload`          | загрузить PDF/TXT и залить в БД             |
| POST   | `/ask`             | ответить на вопрос по документам            |
| POST   | `/ask/stream`      | то же, но ответ потоком (SSE)               |
| GET    | `/documents`       | список документов в базе                    |
| GET    | `/chunks/{index}`  | посмотреть чанк по индексу (для отладки)    |
| GET    | `/health`          | проверка живости                            |
| GET    | `/v1/models`       | список моделей (OpenAI-совместимо)          |

Загрузка идемпотентна: файл с тем же именем заменяет предыдущую версию
(старые чанки удаляются, а не дублируются). Если из файла не удалось
извлечь текст (например, скан без текстового слоя), вернётся `400`.

## Порог релевантности

Если лучший найденный чанк дальше `MAX_DISTANCE`, бот считает, что ответа
в документах нет: LLM не вызывается, ответ приходит мгновенно и без токенов.
Порог задаётся в `.env` (по умолчанию `0.6`).

Подбор порога по реальным числам:

```bash
python eval/calibrate.py                 # вопросы из eval/questions.txt
python eval/calibrate.py my_questions.txt
```

Скрипт печатает `distance` лучшего чанка для каждого вопроса. Порог ставят
в промежутке между числами вопросов, где ответ есть, и вопросов, где его нет.

## Отладка retrieval

Посмотреть, какие чанки нашлись и насколько они близки к вопросу:

```bash
python main.py debug "какая ставка НДФЛ?"
```

Печатает `distance`, источник и начало текста для всех top-4 чанков и помечает,
прошли ли они `MAX_DISTANCE` (`ok` / `far`).

Чтобы видеть то же самое при каждом запросе к API/CLI, добавь в `.env`:

```
DEBUG_RAG=1
```

## Тесты

```bash
pip install -r requirements-dev.txt
pytest          # 25 тестов, ~1 секунда
ruff check .    # линтер
```

Тесты не ходят в сеть и не трогают `chroma_db`: Chroma подменяется на
`EphemeralClient` (в памяти), а вызовы эмбеддингов — на заглушку через
`monkeypatch`. GitHub Actions прогоняет линтер и тесты на каждый push и PR.

## Структура

```
rag/
├── config.py   # клиент OpenRouter, модели, сплиттер, Chroma, MAX_DISTANCE
├── ingest.py   # загрузка документов → чанки → векторы → БД
└── query.py    # retrieval → промпт → LLM → ответ с источниками
main.py         # CLI
api.py          # FastAPI
eval/
├── calibrate.py    # подбор порога релевантности по реальным distance
└── questions.txt   # вопросы для калибровки
static/
├── index.html  # страница (Tailwind CDN, без сборки)
└── app.js      # запросы к API, drag&drop, стриминг ответа
```
