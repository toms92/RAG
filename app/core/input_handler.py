from pydantic import BaseModel, Field
from typing import Any


class PromptRequest(BaseModel):
    """
    Modello per la richiesta in ingresso.
    Gestisce la validazione del JSON ricevuto dal client.
    """
    prompt: str = Field(..., description="Il testo della domanda o del prompt dell'utente", min_length=1)


class RAGResponse(BaseModel):
    """
    Modello per la risposta in uscita.

    - `risposta`: sempre presente, contiene la risposta testuale dell'LLM.
    - `ricette`:  presente SOLO se l'utente ha richiesto ricette ed il Vector DB
                  ha restituito risultati pertinenti.
    """
    risposta: str
    ricette: list[dict[str, Any]] | None = Field(
        default=None,
        description="Ricette strutturate recuperate dal database. Presenti solo se richieste dall'utente."
    )