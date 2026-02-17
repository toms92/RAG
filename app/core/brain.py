from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from core.config import Config

def get_rag_chain(vector_db):
    llm = Ollama(base_url=Config.OLLAMA_URL, model=Config.LLM_MODEL)

    system_prompt = (
        "Sei un assistente culinario preciso. Ragiona bene prima di rispondere. "
        "Usa solo il contesto fornito per consigliare piatti o ingredienti. "
        "Se non lo sai, ammettilo.\n\n"
        "Contesto: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vector_db.as_retriever(), question_answer_chain)