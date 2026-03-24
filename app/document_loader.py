from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_and_chunk_documents(files=["docs/My_GENAI_Projects.docx"]):
    """
    Loads documents from the specified file(s) and chunks them.
    This shared function ensures ingest and retrieval (BM25) have the exact same texts.
    """
    documents = []

    for file in files:
        if not os.path.exists(file):
            print(f"Warning: File {file} not found. Skipping.")
            continue
            
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
    return chunks
