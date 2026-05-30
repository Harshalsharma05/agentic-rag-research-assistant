import os
import arxiv
import requests
from dotenv import load_dotenv
from supabase import create_client
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

# --------------------------------
# Supabase Configuration
# --------------------------------

SUPABASE_URL = require_env("SUPABASE_URL")
SUPABASE_KEY = require_env("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------------------
# Jina Embedding API
# --------------------------------

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v3")
JINA_PASSAGE_TASK = os.getenv("JINA_PASSAGE_TASK", "retrieval.passage")
JINA_EMBEDDING_DIMENSIONS = int(os.getenv("JINA_EMBEDDING_DIMENSIONS", "1024"))


def get_embedding(text: str):

    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": JINA_EMBEDDING_MODEL,
            "task": JINA_PASSAGE_TASK,
            "input": text
        },
        timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Jina API error {response.status_code}: {response.text}"
        )

    embedding = response.json()["data"][0]["embedding"]

    if len(embedding) != JINA_EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Jina embedding dimension mismatch: expected {JINA_EMBEDDING_DIMENSIONS}, got {len(embedding)}"
        )

    return embedding


def embed_chunks(chunks, batch_size=32):

    all_embeddings = []

    for i in range(0, len(chunks), batch_size):

        batch = chunks[i:i + batch_size]

        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": JINA_EMBEDDING_MODEL,
                "task": JINA_PASSAGE_TASK,
                "input": batch
            },
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Jina batch embedding API error {response.status_code}: {response.text}"
            )

        data = response.json()["data"]

        batch_embeddings = [item["embedding"] for item in data]
        bad_embedding = next(
            (embedding for embedding in batch_embeddings if len(embedding) != JINA_EMBEDDING_DIMENSIONS),
            None,
        )
        if bad_embedding is not None:
            raise RuntimeError(
                f"Jina embedding dimension mismatch: expected {JINA_EMBEDDING_DIMENSIONS}, got {len(bad_embedding)}"
            )

        all_embeddings.extend(batch_embeddings)

    return all_embeddings


# --------------------------------
# Ingestion Pipeline
# --------------------------------

def ingest_arxiv_papers(search_query: str, max_results: int = 5, max_papers_to_ingest: int = 2) -> list:
    """Ingest arxiv papers into Supabase. Returns list of paper source strings."""

    def clean_text(value: str) -> str:
        return value.replace("\x00", " ").replace("\u0000", " ")

    try:
        import fitz
    except Exception as e:
        raise RuntimeError(
            "PyMuPDF import failed during paper ingestion."
        ) from e

    print(f"Searching ArXiv for '{search_query}'")

    client = arxiv.Client()

    query_terms = {
        term.lower()
        for term in search_query.split()
        if len(term) > 2
    }

    def score_paper(paper) -> int:
        title = f"{paper.title} {paper.summary}".lower()
        score = 0

        for term in query_terms:
            if term in title:
                score += 1

        # Give extra weight to exact title matches for the most specific terms.
        for term in query_terms:
            if term in paper.title.lower():
                score += 2

        return score

    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = sorted(list(client.results(search)), key=score_paper, reverse=True)

    if not results:
        print("No papers found.")
        return []

    print("[ARXIV RANKING] Top candidates:")
    for index, paper in enumerate(results[:5], start=1):
        print(f"  {index}. score={score_paper(paper)} | {paper.title} ({paper.published.year})")

    if results and score_paper(results[0]) == 0:
        print("[ARXIV WARNING] No strong lexical match found in the top results; using ArXiv relevance order.")

    selected_results = results[:max_papers_to_ingest]
    if len(results) > len(selected_results):
        print(f"[ARXIV LIMIT] Ingesting top {len(selected_results)} papers out of {len(results)} ranked candidates.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    if not os.path.exists("downloads"):
        os.makedirs("downloads")

    paper_sources = []

    for paper in selected_results:

        source_label = f"{paper.title} ({paper.published.year})"

        # Skip if paper already exists in Supabase (prevents duplicate flooding)
        try:
            existing = supabase.table("documents") \
                .select("id") \
                .filter("metadata->>source", "eq", source_label) \
                .limit(1) \
                .execute()
            if existing.data:
                print(f"[SKIP] Already in knowledge base: {paper.title}")
                paper_sources.append(source_label)
                continue
        except Exception as e:
            print(f"[DEDUP WARNING] Could not check duplicates: {e}")

        print(f"\nProcessing: {paper.title}")
        paper_sources.append(source_label)

        pdf_path = f"downloads/{paper.get_short_id()}.pdf"

        print("Downloading PDF...")
        paper.download_pdf(filename=pdf_path)

        print("Extracting text...")

        doc = fitz.open(pdf_path)

        full_text = ""

        for page in doc:
            page_get_text: Any = getattr(page, "get_text", None)
            if callable(page_get_text):
                full_text += str(page_get_text())

        doc.close()

        full_text = clean_text(full_text).replace("\n", " ").strip()

        print("Splitting into chunks...")

        chunks = [clean_text(chunk) for chunk in text_splitter.split_text(full_text)]
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        print(f"{len(chunks)} chunks created")

        metadatas = [
            {"source": source_label}
            for _ in chunks
        ]

        print("Generating embeddings...")

        embeddings = embed_chunks(chunks)
        print(f"[EMBEDDINGS] Generated {len(embeddings)} vectors with dimension {len(embeddings[0]) if embeddings else 0}")

        print("Storing vectors in Supabase...")

        rows = []

        for text, metadata, embedding in zip(chunks, metadatas, embeddings):
            clean_metadata = {}
            for key, value in metadata.items():
                clean_metadata[key] = clean_text(str(value))

            rows.append({
                "content": clean_text(text),
                "metadata": clean_metadata,
                "embedding": embedding
            })

        insert_result = supabase.table("documents").insert(rows).execute()
        print(f"[SUPABASE INSERT] Inserted {len(rows)} rows for {source_label}")
        if getattr(insert_result, "data", None) is not None:
            print(f"[SUPABASE INSERT RESULT] Returned {len(insert_result.data)} rows")

    print("\nKnowledge base updated successfully")
    return paper_sources


# --------------------------------
# Test ingestion
# --------------------------------

if __name__ == "__main__":

    ingest_arxiv_papers(
        "multimodal large language models",
        max_results=1
    )