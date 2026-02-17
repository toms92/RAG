from core.database import get_vector_store
from core.brain import get_rag_chain


def main():
    print("--- Sistema RAG Modulare Avviato ---")

    # Inizializza i componenti
    db = get_vector_store()
    rag = get_rag_chain(db)

    print("\n--- PRONTO PER CHATTARE ---")

    while True:
        user_input = input("Tu: ")
        if user_input.lower() in ["esci", "quit"]: break

        print("Pensando...")
        response = rag.invoke({"input": user_input})
        print(f"\nAI: {response['answer']}\n")


if __name__ == "__main__":
    main()