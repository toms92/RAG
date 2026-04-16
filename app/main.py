from fastapi import FastAPI, HTTPException
import uvicorn

from core.input_handler import PromptRequest, RAGResponse
from core.retriever import RAGRetriever
from core.generator import RAGGenerator

# ---------------------------------------------------------------------------
# Inizializzazione app e componenti
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Nutritional Chatbot API",
    description=(
        "Chatbot nutrizionista con sistema Query Rewriting RAG. "
        "Risponde a domande su alimentazione e benessere. "
        "Se l'utente chiede ricette, interroga il Vector DB e restituisce ricette strutturate."
    ),
    version="2.0.0",
)

try:
    retriever = RAGRetriever()
    generator = RAGGenerator()
    print("Sistema RAG inizializzato con successo.")
except Exception as e:
    print(f"FATAL: Impossibile inizializzare il sistema RAG. Errore: {e}")
    retriever = None
    generator = None

# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/process", response_model=RAGResponse)
async def process_prompt(request: PromptRequest):
    """
    Pipeline condizionale Query Rewriting RAG:

    CASO A — Domanda generale (nessuna ricetta):
        Prompt → LLM → { risposta }

    CASO B — Richiesta di ricette:
        Prompt → detect_recipe_request()
               → rewrite_query()
               → embed(rewritten_query)
               → ChromaDB (top-5, filtro distanza)
               → generate_response(prompt, context)
               → { risposta, ricette }
    """
    if not retriever or not generator:
        raise HTTPException(
            status_code=503,
            detail="Il sistema RAG non è disponibile. Controllare i log del server.",
        )

    try:
        prompt_text = request.prompt
        print(f"\n{'='*50}")
        print(f"Nuova richiesta | Prompt: {prompt_text}")
        print(f"{'='*50}")

        # ----------------------------------------------------------------
        # Step 1: L'LLM classifica se si tratta di una richiesta di ricette
        # ----------------------------------------------------------------
        is_recipe_request = generator.detect_recipe_request(prompt_text)

        if is_recipe_request:
            # ------------------------------------------------------------
            # PIPELINE RAG — attivata solo per le ricette
            # ------------------------------------------------------------
            print("Modalità: RAG (richiesta ricette)")

            # Step 2: Query Rewriting — l'LLM ottimizza la query per il retrieval
            rewritten_query = generator.rewrite_query(prompt_text)

            # Step 3: Retrieval — embedding + ricerca semantica su ChromaDB
            structured_recipes, context_texts = retriever.retrieve_recipes(rewritten_query)

            # Step 4: Generazione risposta testuale (con contesto se disponibile)
            llm_response = generator.generate_response(prompt_text, context_texts or None)

            # Step 5: Risposta finale — include le ricette solo se trovate
            return RAGResponse(
                risposta=llm_response,
                ricette=structured_recipes if structured_recipes else None,
            )

        else:
            # ------------------------------------------------------------
            # PIPELINE DIRETTA — nessun retrieval, solo LLM
            # ------------------------------------------------------------
            print("Modalità: Chatbot diretto (nessuna richiesta di ricette)")

            llm_response = generator.generate_response(prompt_text)

            return RAGResponse(
                risposta=llm_response,
                ricette=None,
            )

    except Exception as e:
        print(f"Errore durante l'elaborazione della richiesta: {e}")
        raise HTTPException(status_code=500, detail=f"Errore interno del server: {e}")

# ---------------------------------------------------------------------------
# Avvio locale
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Avvio del server RAG in modalità sviluppo...")
    uvicorn.run(app, host="0.0.0.0", port=8001)