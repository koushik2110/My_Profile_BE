import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag import ask_rag

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="GenAI RAG API")
logger = logging.getLogger("rag")

class Query(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(query: Query):
    start = time.time()
    logger.info("Ask request received")
    try:
        result = ask_rag(query.question)
        elapsed_ms = int((time.time() - start) * 1000)
        logger.info("Ask request completed in %d ms", elapsed_ms)
        return {
            "answer": result["result"],
            "sources": [doc.page_content[:200] for doc in result["source_documents"]],
        }
    except Exception:
        logger.exception("Ask endpoint failed")
        raise HTTPException(
            status_code=500,
            detail="Ask failed. Check server logs for details.",
        )
