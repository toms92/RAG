import os
import time
from dotenv import load_dotenv

# Import corretti per LangChain 0.2/0.3
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from chromadb import Settings

load_dotenv()

# Carica variabili d'ambiente
load_dotenv()


def main():
    print("--- Avvio Sistema RAG Moderno ---")

    # 1. Inizializzazione Embeddings
    print("Caricamento modello di embedding...")
    # Sostituisci la vecchia inizializzazione con questa
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL_NAME"),
        # Usa HF_HOME se presente, altrimenti TRANSFORMERS_CACHE
        cache_folder=os.getenv("HF_HOME", os.getenv("TRANSFORMERS_CACHE"))
    )

    # 2. Connessione a ChromaDB
    print(f"Connessione a ChromaDB su {os.getenv('CHROMA_HOST')}...")
    # Usiamo la libreria specifica langchain-chroma
    vector_db = Chroma(
        persist_directory="/chroma/chroma",
        embedding_function=embeddings,
        collection_name="recipes_collection",
        client_settings=Settings(anonymized_telemetry=False)  # <--- Aggiungi questo
    )

    # 3. Configurazione Ollama
    print(f"Inizializzazione LLM ({os.getenv('LLM_MODEL')})...")
    llm = Ollama(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        model=os.getenv("LLM_MODEL")
    )

    # 4. Definizione del Prompt e della Catena
    system_prompt = (
        "Sei un assistente culinario, sei preciso e ragioni bene prima di rispondere,"
        " devi consigliare dei piatti all'utente in base alle sue richieste."
        " Usa il contesto per rispondere alla domanda."
        "\n\n"
        "{context}"
    )

    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Dato il messaggio dell'utente, riformulalo per renderlo una query di ricerca efficace per un database di ricette. "
         "Rimuovi convenevoli e focalizzati sugli ingredienti e il tipo di piatto. "
         "Rispondi SOLO con la query riformulata."),
        ("human", "{input}")
    ])

    # Creazione della catena di risposta
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vector_db.as_retriever(), question_answer_chain)

    print("\n--- SISTEMA PRONTO (Scrivi 'esci' per chiudere) ---\n")

    while True:
        user_input = input("Tu: ")
        if user_input.lower() in ["esci", "quit", "exit"]:
            break

        print(f"[*] Analisi della richiesta: {user_input}")

        try:
            # PASSAGGIO A: Riformulazione (Query Transformation)
            # Chiediamo all'LLM di ottimizzare la domanda per Chroma
            optimized_query_chain = rephrase_prompt | llm
            search_query = optimized_query_chain.invoke({"input": user_input})

            print(f"[*] Ricerca nel DB per: {search_query}")

            # PASSAGGIO B: Esecuzione RAG con la query ottimizzata
            # Usiamo 'search_query' per il recupero, ma l'input originale per la risposta
            response = rag_chain.invoke({
                "input": user_input,
                "search_query": search_query  # Dovremo mappare il retriever per usarla
            })

            print(f"\nAI: {response['answer']}\n")


if __name__ == "__main__":
    main()