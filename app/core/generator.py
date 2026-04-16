import requests
import core.config as config


# ---------------------------------------------------------------------------
# System prompt del chatbot nutrizionista
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Sei un medico nutrizionista esperto e dal carattere empatico. "
    "Il tuo scopo è assistere le persone nel loro percorso alimentare, "
    "rispondendo in modo chiaro, professionale e comprensibile. "
    "Rispondi sempre in italiano, indipendentemente dalla lingua dell'utente.\n\n"

    "### RICETTE\n"
    "Se l'utente richiede ricette, fornisci esclusivamente quelle presenti nel database a cui sei collegato, "
    "senza inventare o integrare con informazioni esterne. "
    "Se nel database non sono presenti ricette pertinenti, rispondi esattamente con: "
    "'Mi dispiace, ma non ho trovato ricette adatte nel mio database.'\n\n"

    "### DOMANDE GENERALI\n"
    "Rispondi solo a domande riguardanti alimentazione, nutrizione, diete, ricette, "
    "alimenti, integratori ed esercizio fisico. "
    "Se l'utente fa domande che non riguardano questi ambiti, rispondi esattamente con: "
    "'Posso aiutarti solo su argomenti legati all'alimentazione e al benessere.'\n\n"

    "### REGOLA FONDAMENTALE\n"
    "Non inventare mai informazioni. Se il contesto non contiene la risposta, dillo chiaramente."
)


class RAGGenerator:
    def __init__(self, model_name: str = config.LLM_MODEL, base_url: str = config.OLLAMA_URL):
        self.model_name = model_name
        self.base_url = base_url

    # ------------------------------------------------------------------
    # Metodo privato: unico punto di contatto con Ollama
    # ------------------------------------------------------------------
    def _call_ollama(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        """Metodo helper centralizzato per tutte le chiamate a Ollama."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"Errore durante la chiamata a Ollama: {e}")
            raise

    # ------------------------------------------------------------------
    # Step 1 della pipeline RAG: rilevamento ricette
    # ------------------------------------------------------------------
    def detect_recipe_request(self, prompt: str) -> bool:
        """
        Usa l'LLM per classificare se il prompt contiene una richiesta di ricette.
        Risponde SOLO con 'SI' o 'NO' → parsing deterministico e affidabile.
        """
        system = (
            "Sei un classificatore binario. Rispondi ESCLUSIVAMENTE con 'SI' oppure 'NO', "
            "senza aggiungere nessun altro testo, punteggiatura o spiegazione. "
            "Rispondi 'SI' se il messaggio dell'utente contiene una richiesta di ricette di cucina. "
            "Rispondi 'NO' in tutti gli altri casi."
        )
        result = self._call_ollama(prompt, system=system, temperature=0.0)
        is_recipe = result.strip().upper().startswith("SI")
        label = "RICETTE ✓" if is_recipe else "GENERALE"
        print(f"[Recipe Detection] {label} | Prompt: '{prompt[:60]}...'")
        return is_recipe

    # ------------------------------------------------------------------
    # Step 2 della pipeline RAG: query rewriting
    # ------------------------------------------------------------------
    def rewrite_query(self, prompt: str) -> str:
        """
        Riscrive la query dell'utente per ottimizzarla per la ricerca semantica
        nel Vector DB di ricette culinarie.
        Risponde SOLO con la query riscritta → nessun testo aggiuntivo da gestire.
        """
        system = (
            "Sei un esperto di sistemi di information retrieval applicati alla cucina italiana. "
            "Il tuo unico compito è riscrivere la query dell'utente per massimizzare la precisione "
            "di una ricerca semantica in un database di ricette. "
            "La query riscritta deve essere: specifica, ricca di termini culinari pertinenti e priva di ambiguità. "
            "Rispondi ESCLUSIVAMENTE con la query riscritta, senza spiegazioni o testo aggiuntivo."
        )
        rewritten = self._call_ollama(prompt, system=system, temperature=0.0)
        print(f"[Query Rewriting] '{prompt}' → '{rewritten}'")
        return rewritten

    # ------------------------------------------------------------------
    # Step finale: generazione della risposta testuale
    # ------------------------------------------------------------------
    def generate_response(self, prompt: str, context: list[str] | None = None) -> str:
        """
        Genera la risposta testuale dell'LLM.
        Se viene fornito un contesto (testi delle ricette), lo include nel prompt.
        """
        if context:
            context_str = "\n---\n".join(context)
            user_prompt = f"Contesto fornito:\n{context_str}\n\nDomanda dell'utente:\n{prompt}"
        else:
            user_prompt = prompt

        print(f"Inviando prompt a Ollama | Modello: {self.model_name} | Temperatura: 0.0")
        return self._call_ollama(user_prompt, system=SYSTEM_PROMPT, temperature=0.0)