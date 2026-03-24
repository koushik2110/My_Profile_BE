from langchain_community.vectorstores.pgvector import PGVector, DistanceStrategy
from .document_loader import load_and_chunk_documents
from .fastembed_embeddings import FastEmbedEmbeddings
from .config import DB_URL


chunks = load_and_chunk_documents()


# -------- Embeddings --------
embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# -------- Store in Supabase pgvector --------
PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection_string=DB_URL,
    collection_name="genai_docs_v4",
    distance_strategy=DistanceStrategy.COSINE
)

print("DOCX documents indexed successfully")
