"""Общая конфигурация: клиент OpenRouter, модели, сплиттер, Chroma."""
import os

import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()

# --- LLM-клиент (OpenRouter, OpenAI-совместимый API) ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

CHAT_MODEL = "nvidia/nemotron-3.5-super-vl:free"
EMBED_MODEL = "nvidia/nemotron-3.5-embed-1b:free"

# --- Chunking ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # ~200 токенов: баланс точность/размытость
    chunk_overlap=150,   # чтобы мысль на границе не терялась
    separators=["\n\n", "\n", ". ", " ", ""],
)

# --- Vector DB ---
DB_PATH = "./chroma_db"
COLLECTION_NAME = "docs"

chroma = chromadb.PersistentClient(path=DB_PATH)
collection = chroma.get_or_create_collection(name=COLLECTION_NAME)


def reset_db() -> None:
    """Полностью очищает векторную БД: удаляет коллекцию со всеми чанками и векторами.

    Актуальна только для PersistentClient — EphemeralClient и так умирает с процессом.
    Объект коллекции после удаления «протухает», поэтому пересоздаём и
    подменяем ссылку в этом модуле — все импортировавшие её модули
    обращаются через rag.config.collection, так что увидят новую.
    """
    global collection
    chroma.delete_collection(COLLECTION_NAME)
    collection = chroma.get_or_create_collection(name=COLLECTION_NAME)

