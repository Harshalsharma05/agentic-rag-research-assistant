import os
import arxiv
import requests

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------
# Vector DB path
# --------------------------------

persist_directory = "./chroma_db"

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
    )

    return response.json()["data"][0]["embedding"]


class JinaEmbeddingFunction:

    def embed_documents(self, texts):
        return [get_embedding(text) for text in texts]

    def embed_query(self, text):
        return get_embedding(text)


embeddings = JinaEmbeddingFunction()

# --------------------------------
# Ingestion Pipeline
# --------------------------------

def ingest_arxiv_papers(search_query: str, max_results: int = 1):

    try:
        import fitz
    except Exception as e:
        raise RuntimeError(
            "PyMuPDF import failed during paper ingestion. "
            "Install a compatible pymupdf build for this Python environment."
        ) from e

    print(f"Step 1 & 2: Searching ArXiv for '{search_query}'")

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

    chroma_client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False)
    )
    vector_db = chroma_client.get_or_create_collection("langchain")

    if not os.path.exists("downloads"):
        os.makedirs("downloads")

    for paper in results:

        print(f"\n--- Processing: {paper.title} ---")

        pdf_path = f"downloads/{paper.get_short_id()}.pdf"

        print("Downloading PDF...")
        paper.download_pdf(filename=pdf_path)

        print("Extracting text using PyMuPDF...")

        doc = fitz.open(pdf_path)

        full_text = ""

        for page in doc:
            full_text += page.get_text()

        doc.close()

        print("Cleaning text...")

        full_text = full_text.replace("\n", " ").strip()

        print("Splitting text into chunks...")

        chunks = text_splitter.split_text(full_text)

        print(f"Created {len(chunks)} chunks")

        metadatas = [
            {"source": f"{paper.title} ({paper.published.year})"}
            for _ in chunks
        ]

        print("Generating embeddings and storing in ChromaDB...")

        vector_db.add(
            documents=chunks,
            metadatas=metadatas,
            ids=[f"{paper.get_short_id()}-{index}" for index in range(len(chunks))],
            embeddings=embeddings.embed_documents(chunks),
        )

    print("\nKnowledge base updated successfully")


# --------------------------------
# Test ingestion
# --------------------------------

if __name__ == "__main__":

    ingest_arxiv_papers(
        "multimodal large language models",
        max_results=1
    )