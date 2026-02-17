import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Parametri ChromaDB
    CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
    CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
    CHROMA_COLLECTION = "ricette"
    # Percorso dove Chroma salva i dati nel container
    CHROMA_PERSIST_DIR = "/chroma/chroma"

    # Parametri Modelli di Embedding
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")
    HF_HOME = os.getenv("HF_HOME", "/root/.cache/huggingface")

    # Parametri LLM (Ollama)
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")