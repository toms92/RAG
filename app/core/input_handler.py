from pydantic import BaseModel, Field

class PromptRequest(BaseModel):
    """
    Modello per la richiesta in ingresso.
    Gestisce la validazione del JSON ricevuto dal client.
    """
    prompt: str = Field(..., description="Il testo della domanda o del prompt dell'utente", min_length=1)

class RAGResponse(BaseModel):
    """
    Modello per la risposta in uscita.
    """
    response: str
    context_used: list[str] | None = Field(default=None, description="Opzionale: i frammenti di documenti usati come contesto")
