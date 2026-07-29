# Jina API calls & LangChain Embedding Wrapper
import os
from dotenv import load_dotenv
import requests
from langchain_core.embeddings import Embeddings

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v3")
JINA_PASSAGE_TASK = os.getenv("JINA_PASSAGE_TASK", "retrieval.passage")
JINA_EMBEDDING_DIMENSIONS = int(os.getenv("JINA_EMBEDDING_DIMENSIONS", "1024"))


def get_embedding(text: str, task: str = JINA_PASSAGE_TASK) -> list[float]:
    """
    Fetches a single vector embedding. 
    Defaults to 'retrieval.passage' for ingestion, but allows overrides for queries.
    """
    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": JINA_EMBEDDING_MODEL,
            "task": task, # <--- Now dynamic!
            "input": text,
            "truncate": True,
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Jina API error {response.status_code}: {response.text}")

    embedding = response.json()["data"][0]["embedding"]

    if len(embedding) != JINA_EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Jina embedding dimension mismatch: expected {JINA_EMBEDDING_DIMENSIONS}, got {len(embedding)}"
        )

    return embedding

def embed_chunks(chunks: list[str], batch_size: int = 32) -> list[list[float]]:
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": JINA_EMBEDDING_MODEL,
                "task": JINA_PASSAGE_TASK,
                "input": batch,
                "truncate": True,
            },
            timeout=30,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Jina batch embedding API error {response.status_code}: {response.text}"
            )

        data = response.json()["data"]
        batch_embeddings = [item["embedding"] for item in data]

        for embedding in batch_embeddings:
            if len(embedding) != JINA_EMBEDDING_DIMENSIONS:
                raise RuntimeError(
                    f"Jina embedding dimension mismatch: expected {JINA_EMBEDDING_DIMENSIONS}, got {len(embedding)}"
                )

        all_embeddings.extend(batch_embeddings)

    return all_embeddings


class JinaLangchainEmbeddings(Embeddings):
    """LangChain-compatible wrapper around Jina Batch Embedding API."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_chunks(texts)

    def embed_query(self, text: str) -> list[float]:
        return get_embedding(text)