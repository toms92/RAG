from fastapi import FastAPI, HTTPException
import uvicorn

# Importa i nuovi moduli modulari
from core.input_handler import PromptRequest, RAGResponse
from core.retriever import RAGRetriever
from core.generator import RAGGenerator

# --- Inizializzazione dell'Applicazione e dei Client ---

app = FastAPI(
    title="RAG System API",
    description="API per interrogare un sistema Retrieval-Augmented Generation.",
    version="1.0.0"
)

# Inizializza i componenti principali del sistema RAG
try:
    retriever = RAGRetriever()
    generator = RAGGenerator()
    print("Sistema RAG inizializzato con successo.")
except Exception as e:
    # Se i componenti non si inizializzano, il server non può funzionare.
    # Meglio fermare l'avvio e loggare l'errore.
    print(f"FATAL: Impossibile inizializzare il sistema RAG. Errore: {e}")
    retriever = None
    generator = None

# --- Definizione degli Endpoint ---

@app.post("/process", response_model=RAGResponse)
async def process_prompt(request: PromptRequest):
    """
    Endpoint principale per elaborare una richiesta RAG.
    
    1. Riceve un prompt.
    2. Recupera il contesto rilevante da ChromaDB.
    3. Genera una risposta basata sul prompt e sul contesto.
    """
    if not retriever or not generator:
        raise HTTPException(status_code=503, detail="Il sistema RAG non è disponibile. Controllare i log del server.")

    try:
        prompt_text = request.prompt
        print(f"--- Nuova Richiesta Ricevuta ---")
        print(f"Prompt: {prompt_text}")

        # 1. Fase di Recupero (Retrieve)
        print("Fase 1: Recupero del contesto...")
        retrieved_context = retriever.retrieve_context(prompt_text)
        print(f"Contesto recuperato: {len(retrieved_context)} documenti.")

        # 2. Fase di Generazione (Generate)
        print("Fase 2: Generazione della risposta...")
        llm_response = generator.generate_response(prompt_text, retrieved_context)
        print(f"Risposta generata: {llm_response[:100]}...") # Logga solo l'inizio

        # 3. Restituzione della risposta
        return RAGResponse(response=llm_response, context_used=retrieved_context)

    except Exception as e:
        print(f"Errore durante l'elaborazione della richiesta: {e}")
        raise HTTPException(status_code=500, detail=f"Errore interno del server: {e}")

# --- Avvio del Server (per esecuzione locale) ---

if __name__ == "__main__":
    print("Avvio del server RAG in modalità sviluppo...")
    uvicorn.run(app, host="0.0.0.0", port=5000)
