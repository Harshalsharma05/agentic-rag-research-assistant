# Clean Ingestion Workflow & Supabase Orchestration
import os
import time
from typing import Any
from dotenv import load_dotenv
from supabase import create_client

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

# Import modularized components
from backend.services.embeddings import embed_chunks, JinaLangchainEmbeddings
from backend.services.academic_search import get_papers_from_semantic_scholar, download_pdf_bytes

load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


SUPABASE_URL = require_env("SUPABASE_URL")
SUPABASE_KEY = require_env("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def clean_text(value: str) -> str:
    return value.replace("\x00", " ").replace("\u0000", " ")


def ingest_arxiv_papers(search_query: str, max_results: int = 10, max_papers_to_ingest: int = 2) -> list[str]:
    """Ingest papers into Supabase with automatic candidate fallback and race-condition control."""
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError("PyMuPDF import failed during paper ingestion.") from e

    print(f"Searching Semantic Scholar for '{search_query}'")
    candidates = get_papers_from_semantic_scholar(search_query, limit=max_results)

    if not candidates:
        print("No valid paper candidates found.")
        return []

    # Initialize Chunkers
    jina_embedder = JinaLangchainEmbeddings()
    semantic_splitter = SemanticChunker(jina_embedder, breakpoint_threshold_type="percentile")
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)

    if not os.path.exists("downloads"):
        os.makedirs("downloads")

    paper_sources = []
    successful_ingested = 0

    # RESILIENT FALLBACK LOOP: Iterate through sorted candidates until quota is reached
    for paper in candidates:
        if successful_ingested >= max_papers_to_ingest:
            break

        source_label = f"{paper['title']} ({paper['year']})"
        unique_paper_id = paper.get("arxiv_id") or paper["title"].replace(" ", "_")
        db_paper_id = None

        print(f"\nProcessing Candidate: {paper['title']}")

        # 1. ATOMIC REGISTRY CHECK
        try:
            paper_registry = supabase.table("papers").insert({
                "paper_id": unique_paper_id,
                "title": paper["title"],
                "pdf_url": paper["pdf_url"],
                "status": "processing"
            }).execute()

            registry_data = getattr(paper_registry, "data", [])
            
            if not registry_data or not isinstance(registry_data, list):
                raise RuntimeError("Invalid response from Supabase registry insert.")
                
            db_paper_id = registry_data[0].get("id")
            
            if not db_paper_id:
                raise RuntimeError("Supabase returned data without an ID.")

            print("[REGISTRY] Claimed paper for ingestion.")

        except Exception:
            print(f"[RACE CONDITION HANDLED] Paper '{paper['title']}' is already processing or completed.")
            
            # Polling wait for concurrent threads
            for _ in range(15):
                existing = supabase.table("papers").select("id", "status").eq("paper_id", unique_paper_id).execute()
                if existing.data and existing.data[0]["status"] == "completed":
                    print("[SKIP] Paper fully ingested by another thread.")
                    break
                time.sleep(2)

            paper_sources.append(source_label)
            successful_ingested += 1
            continue

        # 2. DOWNLOAD & PARSE WITH RESILIENT CATCH
        try:
            print(f"Downloading PDF from {paper['pdf_url']}...")
            pdf_bytes = download_pdf_bytes(paper["pdf_url"])

            pdf_path = f"downloads/{unique_paper_id}.pdf"
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)

            print("Extracting text via PyMuPDF...")
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                page_get_text: Any = getattr(page, "get_text", None)
                if callable(page_get_text):
                    full_text += str(page_get_text())
            doc.close()

            full_text = clean_text(full_text).replace("\n", " ").strip()
            if not full_text:
                raise RuntimeError("Extracted PDF text was empty.")

            # 3. PARENT-CHILD HIERARCHICAL INGESTION
            print("Generating Parent-Child hierarchies...")
            parent_blocks = parent_splitter.split_text(full_text)

            parent_rows = [
                {
                    "paper_id": db_paper_id,
                    "content": clean_text(block).strip(),
                    "metadata": {"source": source_label, "type": "parent"}
                }
                for block in parent_blocks if clean_text(block).strip()
            ]

            print(f"Inserting {len(parent_rows)} Parent chunks...")
            parent_data = getattr(supabase.table("documents").insert(parent_rows).execute(), "data", [])

            child_rows, all_child_texts, child_parent_mapping = [], [], []

            print("Semantically splitting Parents into Children...")
            for parent_row in parent_data:
                parent_uuid = parent_row["id"]
                semantic_docs = semantic_splitter.create_documents([parent_row["content"]])

                for doc in semantic_docs:
                    clean_child = clean_text(doc.page_content).strip()
                    if clean_child:
                        all_child_texts.append(clean_child)
                        child_parent_mapping.append(parent_uuid)

            print(f"Generating {len(all_child_texts)} Jina embeddings...")
            child_embeddings = embed_chunks(all_child_texts)

            for text, p_uuid, embedding in zip(all_child_texts, child_parent_mapping, child_embeddings):
                child_rows.append({
                    "paper_id": db_paper_id,
                    "parent_id": p_uuid,
                    "content": text,
                    "embedding": embedding,
                    "metadata": {"source": source_label, "type": "child"}
                })

            print(f"Inserting {len(child_rows)} linked Child chunks...")
            supabase.table("documents").insert(child_rows).execute()

            # Mark complete
            supabase.table("papers").update({"status": "completed"}).eq("id", db_paper_id).execute()
            print(f"[REGISTRY] Marked {source_label} as completed.")

            paper_sources.append(source_label)
            successful_ingested += 1

        except Exception as e:
            print(f"[INGEST FAILED] Skipping candidate '{paper['title']}': {str(e)}")
            # Mark status as failed in DB so polling loops don't hang
            if db_paper_id:
                supabase.table("papers").update({"status": "failed"}).eq("id", db_paper_id).execute()
            continue

    print(f"\nKnowledge base updated successfully with {successful_ingested} paper(s).")
    return paper_sources


if __name__ == "__main__":
    ingest_arxiv_papers("multimodal large language models", max_results=5, max_papers_to_ingest=1)