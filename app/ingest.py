from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from app.config import DB_URL

files = [
    "docs/My_GENAI_Projects.docx"
]

documents = []

for file in files:
    loader = Docx2txtLoader(file)
    docs = loader.load()

    # add metadata so we know source project
    for d in docs:
        d.metadata["project"] = file

    documents.extend(docs)

# -------- Chunking --------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=80,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_documents(documents)

# -------- Embeddings --------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------- Store in Supabase pgvector --------
PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection_string=DB_URL,
    collection_name="genai_docs"
)

print("✅ DOCX documents indexed successfully")