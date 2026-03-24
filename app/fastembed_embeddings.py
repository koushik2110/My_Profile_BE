from __future__ import annotations

from typing import List

from fastembed import TextEmbedding


class FastEmbedEmbeddings:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        # Initialize the actual embedding model that will run locally to turn text into vectors
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [
            emb.tolist() if hasattr(emb, "tolist") else list(emb)
            for emb in self.model.embed(texts)
        ]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
