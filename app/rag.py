from functools import lru_cache
import logging
import time

from langchain_community.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from .config import DB_URL, GROQ_API_KEY, SYSTEM_PROMPT
from .fastembed_embeddings import FastEmbedEmbeddings
from .document_loader import load_and_chunk_documents

logger = logging.getLogger("rag")


def _require_env() -> None:
    missing = []
    if not DB_URL:
        missing.append("SUPABASE_DB_URL")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


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
    _require_env()
    start = time.time()
    logger.info("Initializing embeddings")
    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # 2. Connect to the PostgreSQL vector database where our document chunks are stored
    logger.info("Initializing vector store")
    vectorstore = PGVector(
        connection_string=DB_URL,
        embedding_function=embeddings,  # Tells the DB how to process incoming query strings
        collection_name="genai_docs_v4",
        distance_strategy=DistanceStrategy.COSINE # Explicitly tell PGVector to use Cosine Similarity
    )

    # 3. Create a standard retriever that fetches the top 5 most relevant chunks regardless of tight score math
    pg_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    logger.info("Initializing BM25 retriever")
    chunks = load_and_chunk_documents()
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10
    
    logger.info("Initializing Ensemble retriever")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, pg_retriever],
        weights=[0.5, 0.5]
    )

    logger.info("Initializing Reranker (Flashrank)")
    compressor = FlashrankRerank()
    
    # Wrap the ensemble retriever with the reranker
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )

    logger.info("Initializing Groq LLM client")
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=compression_retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": _build_prompt()},
        return_source_documents=True
    )
    elapsed_ms = int((time.time() - start) * 1000)
    logger.info("RAG chain initialized in %d ms", elapsed_ms)
    return chain

def ask_rag(question: str):
    return _get_qa_chain()({"query": question})
