import os
import arxiv
import requests
from supabase import create_client

from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------
# Supabase Configuration
# --------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------------------
# Jina Embedding API
# --------------------------------

JINA_API_KEY = os.getenv("JINA_API_KEY")


def get_embedding(text: str):

    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "jina-embeddings-v2-base-en",
            "input": text
        },
        timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Jina API error {response.status_code}: {response.text}"
        )

    return response.json()["data"][0]["embedding"]


def embed_chunks(chunks):
    return [get_embedding(chunk) for chunk in chunks]


# --------------------------------
# Ingestion Pipeline
# --------------------------------

def ingest_arxiv_papers(search_query: str, max_results: int = 1):

    try:
        import fitz
    except Exception as e:
        raise RuntimeError(
            "PyMuPDF import failed during paper ingestion."
        ) from e

    print(f"Searching ArXiv for '{search_query}'")

    client = arxiv.Client()

    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = list(client.results(search))

    if not results:
        print("No papers found.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    if not os.path.exists("downloads"):
        os.makedirs("downloads")

    for paper in results:

        print(f"\nProcessing: {paper.title}")

        pdf_path = f"downloads/{paper.get_short_id()}.pdf"

        print("Downloading PDF...")
        paper.download_pdf(filename=pdf_path)

        print("Extracting text...")

        doc = fitz.open(pdf_path)

        full_text = ""

        for page in doc:
            full_text += page.get_text()

        doc.close()

        full_text = full_text.replace("\n", " ").strip()

        print("Splitting into chunks...")

        chunks = text_splitter.split_text(full_text)

        print(f"{len(chunks)} chunks created")

        metadatas = [
            {"source": f"{paper.title} ({paper.published.year})"}
            for _ in chunks
        ]

        print("Generating embeddings...")

        embeddings = embed_chunks(chunks)

        print("Storing vectors in Supabase...")

        rows = []

        for text, metadata, embedding in zip(chunks, metadatas, embeddings):

            rows.append({
                "content": text,
                "metadata": metadata,
                "embedding": embedding
            })

        supabase.table("documents").insert(rows).execute()

    print("\nKnowledge base updated successfully")


# --------------------------------
# Test ingestion
# --------------------------------

if __name__ == "__main__":

    ingest_arxiv_papers(
        "multimodal large language models",
        max_results=1
    )