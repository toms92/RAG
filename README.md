# Sistema RAG Modulare con Docker e FastAPI

## 1. Panoramica

Questo progetto implementa un sistema RAG (Retrieval-Augmented Generation) modulare e containerizzato. L'architettura è stata progettata per essere scalabile e facile da mantenere, separando chiaramente le responsabilità di gestione dell'input, recupero delle informazioni (Retrieval) e generazione delle risposte (Generation).

## 2. Architettura dei Microservizi

Il sistema è orchestrato tramite Docker Compose e comprende i seguenti servizi:

- **`rag` (App Core)**: Il cuore del sistema. Un'API FastAPI che orchestra il flusso RAG utilizzando moduli Python specializzati.
- **`chroma` (Vector DB)**: Database vettoriale per l'indicizzazione e la ricerca semantica dei documenti.
- **`ollama` (LLM)**: Motore di inferenza per il Large Language Model (es. Llama 3.2).
- **`jupyter`**: Ambiente di sviluppo e analisi dati.

### 2.1 Struttura Modulare del Codice (`app/core`)

Il codice applicativo è stato rifattorizzato in moduli specifici all'interno di `app/core`:

- **`input_handler.py`**: Definisce i modelli Pydantic per la validazione delle richieste e delle risposte API. Assicura che i dati in ingresso siano corretti prima di essere elaborati.
- **`retriever.py`**: Gestisce l'interazione con il database vettoriale. Si occupa di trasformare il prompt utente in un embedding (vettore numerico) e di interrogare ChromaDB per trovare i documenti più rilevanti.
- **`generator.py`**: Gestisce l'interazione con l'LLM (Ollama). Costruisce il prompt finale arricchito con il contesto recuperato e invia la richiesta al modello per ottenere la risposta.
- **`config.py`**: Centralizza la gestione delle configurazioni, leggendo le variabili d'ambiente dal file `.env`.

## 3. Prerequisiti

- **Docker** e **Docker Compose** installati.
- Un file `.env` configurato nella root del progetto (vedi sezione Configurazione).

## 4. Configurazione (.env)

Assicurati che il file `.env` contenga le seguenti configurazioni:

```env
# --- CHROMA DB ---
CHROMA_HOST=chroma
CHROMA_PORT=8000

# --- EMBEDDINGS ---
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
HF_HOME=/root/.cache/huggingface

# --- LLM (OLLAMA) ---
OLLAMA_BASE_URL=http://ollama:11434
LLM_MODEL=llama3.2:3b

# --- API ---
APP_PORT=8001
```

## 5. Avvio e Utilizzo

### 5.1 Avvio del Sistema

Per avviare l'intera infrastruttura:

```sh
docker-compose up --build
```

Questo comando costruirà le immagini Docker (se necessario) e avvierà tutti i servizi.

### 5.2 Utilizzo dell'API

Una volta avviato, il servizio RAG è in ascolto sulla porta `8001`.

**Endpoint**: `POST http://localhost:8001/process`

**Esempio di richiesta (cURL):**

```bash
curl -X POST "http://localhost:8001/process" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "Come si prepara la carbonara?"
         }'
```

**Esempio di Risposta:**

```json
{
  "response": "Per preparare la carbonara hai bisogno di guanciale, pecorino, uova e pepe...",
  "context_used": [
    "La carbonara è un piatto tipico romano...",
    "Ingredienti: Guanciale, Pecorino Romano, Uova..."
  ]
}
```

La risposta include ora anche il campo `context_used`, che mostra quali frammenti di testo sono stati recuperati dal database e utilizzati per generare la risposta, aumentando la trasparenza del sistema.

## 6. Sviluppo e Manutenzione

La nuova struttura modulare facilita lo sviluppo:

- Se devi cambiare il modo in cui i dati vengono recuperati, modifica solo `core/retriever.py`.
- Se vuoi cambiare il prompt di sistema inviato all'LLM, modifica `core/generator.py`.
- Se vuoi aggiungere nuovi campi all'API, modifica `core/input_handler.py`.

### Comandi Utili

- **Log in tempo reale**: `docker-compose logs -f rag`
- **Arresto**: `docker-compose down`
