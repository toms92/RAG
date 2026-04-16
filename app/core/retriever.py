import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import core.config as config


class RAGRetriever:
    def __init__(self):
        # Inizializza il modello di embedding
        try:
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            print(f"Modello di embedding '{config.EMBEDDING_MODEL}' caricato con successo.")
        except Exception as e:
            print(f"Errore nel caricamento del modello di embedding: {e}")
            raise

        # Inizializza il client ChromaDB
        try:
            self.client = chromadb.HttpClient(
                host=config.CHROMA_HOST,
                port=config.CHROMA_PORT,
                settings=Settings(allow_reset=True, anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(name=config.CHROMA_COLLECTION)
            print(f"Connesso a ChromaDB su {config.CHROMA_HOST}:{config.CHROMA_PORT}")
        except Exception as e:
            print(f"Errore nella connessione a ChromaDB: {e}")
            raise

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def embed_prompt(self, prompt: str):
        """Genera l'embedding per un dato testo."""
        return self.embedding_model.encode(prompt)

    # ------------------------------------------------------------------
    # Parsing e pulizia delle ricette
    # ------------------------------------------------------------------
    def _parse_recipe(self, doc: str, metadata: dict | None = None) -> dict | None:
        """
        Tenta di estrarre una ricetta strutturata dal documento o dai suoi metadati.

        Strategia:
          1. Prova a parsare il documento come stringa JSON.
          2. Se fallisce, prova a costruire la ricetta dai metadati ChromaDB.
          3. Se entrambi falliscono, restituisce None (il doc viene usato solo come contesto testuale).
        """
        # 1. Il documento è una stringa JSON?
        try:
            parsed = json.loads(doc)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        # 2. I metadati contengono campi strutturati?
        if metadata:
            recipe: dict = {}
            for key, value in metadata.items():
                if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                    try:
                        recipe[key] = json.loads(value)
                        continue
                    except json.JSONDecodeError:
                        pass
                recipe[key] = value
            if recipe:
                return recipe

        return None

    def _clean_recipe(self, recipe: dict) -> dict:
        """
        Rimuove ricorsivamente i campi nulli, vuoti o assenti dalla ricetta,
        come richiesto dal GEMINI.md (nessun errore, omissione silenziosa).
        """
        cleaned: dict = {}
        for key, value in recipe.items():
            if value is None or value == "" or value == [] or value == {}:
                continue
            if isinstance(value, dict):
                nested = self._clean_recipe(value)
                if nested:
                    cleaned[key] = nested
            elif isinstance(value, list):
                filtered_list = []
                for item in value:
                    if item is None or item == "":
                        continue
                    filtered_list.append(self._clean_recipe(item) if isinstance(item, dict) else item)
                if filtered_list:
                    cleaned[key] = filtered_list
            else:
                cleaned[key] = value
        return cleaned

    # ------------------------------------------------------------------
    # Metodo principale di retrieval (usato dalla pipeline RAG)
    # ------------------------------------------------------------------
    def retrieve_recipes(
        self,
        query: str,
        n_results: int = 5,
        distance_threshold: float = 1.2,
    ) -> tuple[list[dict], list[str]]:
        """
        Dato un query (già riscritto dall'LLM), genera l'embedding e recupera
        le ricette più simili da ChromaDB.

        Returns:
            structured_recipes: lista di dict ricetta puliti (per il campo `ricette` dell'output)
            context_texts:      lista di testi grezzi (per il contesto dell'LLM)
        """
        query_embedding = self.embed_prompt(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )

        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        structured_recipes: list[dict] = []
        context_texts: list[str] = []

        print("\n--- Analisi Punteggi di Similarità (ChromaDB) ---")

        for doc, dist, meta in zip(documents, distances, metadatas):
            preview = doc[:60].replace("\n", " ") + "..."

            if dist <= distance_threshold:
                print(f"[ACCETTATO] Distanza: {dist:.4f} | {preview}")

                # Accumula il testo grezzo per il contesto dell'LLM
                context_texts.append(doc)

                # Tenta di estrarre la ricetta strutturata
                recipe = self._parse_recipe(doc, meta)
                if recipe:
                    structured_recipes.append(self._clean_recipe(recipe))
            else:
                print(f"[SCARTATO]  Distanza: {dist:.4f} | {preview}")

        print(
            f"-> Ricette strutturate: {len(structured_recipes)} | "
            f"Testi contesto: {len(context_texts)} | "
            f"su {len(documents)} estratti totali.\n"
        )

        return structured_recipes, context_texts

    # ------------------------------------------------------------------
    # Metodo legacy mantenuto per compatibilità (non usato dalla nuova pipeline)
    # ------------------------------------------------------------------
    def retrieve_context(
        self,
        prompt: str,
        n_results: int = 5,
        distance_threshold: float = 1.2,
    ) -> list[str]:
        """Mantiene la compatibilità con il codice precedente."""
        _, context_texts = self.retrieve_recipes(prompt, n_results, distance_threshold)
        return context_texts