from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from core.config import Config


def get_vector_store():
    print(f"[*] Connessione a ChromaDB ({Config.CHROMA_COLLECTION})...")

    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        cache_folder=Config.HF_HOME
    )

    http_client = chromadb.HttpClient(
        host=Config.CHROMA_HOST,
        port=int(Config.CHROMA_PORT),
    )

    vector_db = Chroma(
        client=http_client,
        embedding_function=embeddings,
        collection_name=Config.CHROMA_COLLECTION,
    )

    return vector_db