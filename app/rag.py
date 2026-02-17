from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from app.config import DB_URL, GROQ_API_KEY, SYSTEM_PROMPT


def _build_prompt() -> PromptTemplate:
    system_prefix = SYSTEM_PROMPT.strip()
    if system_prefix:
        system_prefix = system_prefix + "\n\n"

    return PromptTemplate.from_template(
        f"{system_prefix}"
        "Use the following context to answer the question.\n"
        "If you don't know the answer, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )


@lru_cache(maxsize=1)
def _get_qa_chain() -> RetrievalQA:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = PGVector(
        connection_string=DB_URL,
        embedding_function=embeddings,
        collection_name="genai_docs"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": _build_prompt()},
        return_source_documents=True
    )

def ask_rag(question: str):
    return _get_qa_chain()({"query": question})
