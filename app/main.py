from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import ask_rag

app = FastAPI(title="GenAI RAG API")

class Query(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/ask")
def ask(query: Query):
    result = ask_rag(query.question)
    return {
        "answer": result["result"],
        "sources": [doc.page_content[:200] for doc in result["source_documents"]]
    }
