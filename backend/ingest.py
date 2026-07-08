import os
import time
import arxiv
import requests
from dotenv import load_dotenv
from supabase import create_client
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()

# Environment and Configuration
def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

SUPABASE_URL = require_env("SUPABASE_URL")
SUPABASE_KEY = require_env("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v3")
JINA_PASSAGE_TASK = os.getenv("JINA_PASSAGE_TASK", "retrieval.passage")
JINA_EMBEDDING_DIMENSIONS = int(os.getenv("JINA_EMBEDDING_DIMENSIONS", "1024"))

SEMANTIC_SCHOLAR_API_KEY = require_env("SEMANTIC_SCHOLAR_API_KEY")

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
            "input": text,
            "truncate": True
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
                "input": batch,
                "truncate": True
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


# Ingestion Pipeline

def search_semantic_scholar(query: str, limit: int = 10):

    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    response = requests.get(
        url,
        headers={
            "x-api-key": SEMANTIC_SCHOLAR_API_KEY
        },
        params={
            "query": query,
            "limit": limit,
            "fields": "title,year,citationCount,externalIds,url,openAccessPdf"
        },
        timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Semantic Scholar API error {response.status_code}: {response.text}"
        )

    return response.json().get("data", [])

def get_papers_from_semantic_scholar(
    search_query: str,
    limit: int = 5
):

    results = search_semantic_scholar(
        search_query,
        limit=limit
    )

    if not results:
        return []

    print("\n[SEMANTIC SCHOLAR RANKING]")

    papers = []
    seen_titles = set()

    for index, result in enumerate(results, start=1):

        title = result.get("title", "")
        year = result.get("year")
        citations = result.get("citationCount", 0)

        arxiv_id = (
            result.get("externalIds", {})
            .get("ArXiv")
        )

        pdf_url = None

        if arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        elif result.get("openAccessPdf"):
            pdf_url = result["openAccessPdf"].get("url")

        print(
            f"{index}. citations={citations} | "
            f"{title} ({year}) | "
            f"pdf={pdf_url}"
        )

        if not pdf_url:
            continue
        
        normalized_title = title.lower().strip()

        if normalized_title in seen_titles:
            continue

        seen_titles.add(normalized_title)

        papers.append({
            "title": title,
            "year": year,
            "pdf_url": pdf_url,
            "arxiv_id": arxiv_id
        })

    return papers


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

    print(
        f"Searching Semantic Scholar for "
        f"'{search_query}'"
    )

    results = get_papers_from_semantic_scholar(
        search_query,
        limit=max_results
    )

    if not results:
        print("No papers found.")
        return []
    

    selected_results = results[:max_papers_to_ingest]
    
    if len(results) > len(selected_results):
        print(f"[ARXIV LIMIT] Ingesting top {len(selected_results)} papers out of {len(results)} ranked candidates.")

    jina_embedder = JinaLangchainEmbeddings()
    semantic_splitter = SemanticChunker(
        jina_embedder, breakpoint_threshold_type="percentile"
    )

    if not os.path.exists("downloads"):
        os.makedirs("downloads")

    paper_sources = []

    for paper in selected_results:

        source_label = f"{paper['title']} ({paper['year']})"

        # Skip if paper already exists in Supabase (prevents duplicate flooding)
        print(f"\nProcessing: {paper['title']}")
        
        unique_paper_id = paper.get("arxiv_id") or paper["title"].replace(" ", "_")
        db_paper_id = None
        
        # 1. ATOMIC REGISTRY CHECK (The Gatekeeper)
        try:
            # Attempt to insert and claim the paper
            paper_registry = supabase.table("papers").insert({
                "paper_id": unique_paper_id,
                "title": paper["title"],
                "pdf_url": paper["pdf_url"],
                "status": "processing"
            }).execute()
            
            db_paper_id = paper_registry.data[0]["id"]
            print("[REGISTRY] Claimed new paper for ingestion.")
            
        except Exception as e:
            # Conflict triggered! Postgres blocked a duplicate insert.
            print(f"[RACE CONDITION HANDLED] Paper '{paper['title']}' is already processing or completed.")
            
            # Enter polling loop for User B
            max_retries = 30 # Wait up to 30 seconds
            for attempt in range(max_retries):
                existing = supabase.table("papers").select("id", "status").eq("paper_id", unique_paper_id).execute()
                
                if existing.data:
                    current_status = existing.data[0]["status"]
                    if current_status == "completed":
                        print(f"[SKIP] Paper fully ingested by another thread. Ready for search.")
                        break
                    else:
                        print("Another agent is currently embedding this paper. Waiting 2s...")
                        time.sleep(2)
                else:
                    break
            
            paper_sources.append(source_label)
            continue # Skip the rest of the loop (do not download/embed)

        paper_sources.append(source_label)
        paper_id = unique_paper_id # Fallback for your local file saving logic

        pdf_path = f"downloads/{paper_id}.pdf"

        print("Downloading PDF...")

        pdf_response = requests.get(
            paper["pdf_url"],
            timeout=60
        )

        pdf_response.raise_for_status()

        with open(pdf_path, "wb") as f:
            f.write(pdf_response.content)

        print("Extracting text...")

        doc = fitz.open(pdf_path)

        full_text = ""

        for page in doc:
            page_get_text: Any = getattr(page, "get_text", None)
            if callable(page_get_text):
                full_text += str(page_get_text())

        doc.close()

        full_text = clean_text(full_text).replace("\n", " ").strip()

        print("Generating Parent-Child hierarchies...")
        
        # 1. GENERATE PARENTS (Broad structural blocks for Gemini)
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
        parent_blocks = parent_splitter.split_text(full_text)
        
        parent_rows = []
        for block in parent_blocks:
            clean_block = clean_text(block).strip()
            if clean_block:
                parent_rows.append({
                    "paper_id": db_paper_id,
                    "content": clean_block,
                    "metadata": {"source": source_label, "type": "parent"}
                })
                
        print(f"Inserting {len(parent_rows)} Parent chunks...")
        # Insert and return rows so we have the Parent UUIDs
        parent_insert_result = supabase.table("documents").insert(parent_rows).execute()
        parent_data = getattr(parent_insert_result, "data", [])
        
        # 2. GENERATE CHILDREN (Semantic blocks for Jina Vector Search)
        child_rows = []
        all_child_texts = []
        child_parent_mapping = [] 
        
        print("Semantically splitting Parents into Children...")
        for parent_row in parent_data:
            parent_uuid = parent_row["id"]
            parent_text = parent_row["content"]
            
            # Semantic split this specific parent
            semantic_docs = semantic_splitter.create_documents([parent_text])
            
            for doc in semantic_docs:
                clean_child = clean_text(doc.page_content).strip()
                if clean_child:
                    all_child_texts.append(clean_child)
                    child_parent_mapping.append(parent_uuid) # Track which parent this belongs to
                    
        print(f"Generating {len(all_child_texts)} Jina embeddings for Children...")
        child_embeddings = embed_chunks(all_child_texts)
        
        # 3. LINK AND INSERT CHILDREN
        for text, p_uuid, embedding in zip(all_child_texts, child_parent_mapping, child_embeddings):
            child_rows.append({
                "paper_id": db_paper_id,
                "parent_id": p_uuid, # <-- Explicit relational link
                "content": text,
                "embedding": embedding,
                "metadata": {"source": source_label, "type": "child"}
            })
            
        print(f"Inserting {len(child_rows)} linked Child chunks...")
        child_insert_result = supabase.table("documents").insert(child_rows).execute()
        if getattr(child_insert_result, "data", None) is not None:
            print(f"[SUPABASE INSERT RESULT] Returned {len(child_insert_result.data)} rows")
            
        supabase.table("papers").update({"status": "completed"}).eq("id", db_paper_id).execute()
        print(f"[REGISTRY] Marked {source_label} as completed. Unlocked for search.")

    print("\nKnowledge base updated successfully")
    return paper_sources


class JinaLangchainEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Reuses your highly efficient batching function!
        return embed_chunks(texts)
    
    def embed_query(self, text: str) -> list[float]:
        return get_embedding(text)

# Test ingestion
if __name__ == "__main__":

    ingest_arxiv_papers(
        "multimodal large language models",
        max_results=1
    )