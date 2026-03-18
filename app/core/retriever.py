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
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(name=config.CHROMA_COLLECTION)
            print(f"Connesso a ChromaDB su {config.CHROMA_HOST}:{config.CHROMA_PORT}")
        except Exception as e:
            print(f"Errore nella connessione a ChromaDB: {e}")
            raise

    def embed_prompt(self, prompt: str):
        """Genera l'embedding per un dato testo."""
        return self.embedding_model.encode(prompt)

    # --- AGGIUNTA SOGLIA DI DISTANZA (distance_threshold) ---
    def retrieve_context(self, prompt: str, n_results: int = 5, distance_threshold: float = 1.2):
        """
        Dato un prompt, genera l'embedding e recupera i documenti più simili da ChromaDB.
        Filtra i risultati scartando quelli con una distanza superiore alla soglia.
        """
        prompt_embedding = self.embed_prompt(prompt)

        # Chroma restituisce un dizionario con 'documents', 'distances', 'ids', ecc.
        results = self.collection.query(
            query_embeddings=[prompt_embedding.tolist()],
            n_results=n_results
        )

        # Estraiamo le liste dei documenti e delle distanze
        documents = results.get('documents', [[]])[0]
        distances = results.get('distances', [[]])[0]

        filtered_documents = []

        print("\n--- Analisi Punteggi di Similarità (ChromaDB) ---")

        # Iteriamo contemporaneamente sui documenti e sulle loro distanze
        for doc, dist in zip(documents, distances):
            # Stampiamo i primi 50 caratteri del documento per capire di cosa parla
            preview = doc[:50].replace("\n", " ") + "..."

            # Se la distanza è minore o uguale alla soglia, il documento è pertinente
            if dist <= distance_threshold:
                print(f"[ACCETTATO] Distanza: {dist:.4f} | {preview}")
                filtered_documents.append(doc)
            else:
                print(f"[SCARTATO]  Distanza: {dist:.4f} | {preview}")

        print(f"-> Contesto finale: {len(filtered_documents)} documenti validi su {len(documents)} estratti.\n")

        return filtered_documents