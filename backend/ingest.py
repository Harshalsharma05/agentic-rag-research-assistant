import os
import arxiv
import fitz  # This is PyMuPDF
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setup DB path (Same as in main.py)
persist_directory = "./chroma_db"

def ingest_arxiv_papers(search_query: str, max_results: int = 1):
    print(f"Step 1 & 2: Searching ArXiv for '{search_query}' and retrieving metadata...")
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

    # Setup Embedding Model and Text Splitter
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # PDF Step 6 Setup: Split into chunks of 1000 characters with 100 char overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    # Load our existing Vector DB
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Create a folder to store downloaded PDFs
    if not os.path.exists("downloads"):
        os.makedirs("downloads")

    for paper in results:
        print(f"\n--- Processing: {paper.title} ---")
        
        # Step 3: Download Research Papers
        pdf_path = f"downloads/{paper.get_short_id()}.pdf"
        print(f"Step 3: Downloading PDF to {pdf_path}...")
        paper.download_pdf(filename=pdf_path)

        # Step 4: Extract Text from PDFs
        print("Step 4: Extracting text using PyMuPDF...")
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()
        
        # Step 5: Clean and Preprocess Text
        print("Step 5: Cleaning text...")
        full_text = full_text.replace('\n', ' ').strip()

        # Step 6: Split Text into Chunks
        print("Step 6: Splitting text into chunks...")
        chunks = text_splitter.split_text(full_text)
        print(f"Created {len(chunks)} chunks.")

        # Prepare metadata (Citations)
        metadatas =[{"source": f"{paper.title} ({paper.published.year})"} for _ in chunks]

        # Step 7 & 8: Generate Embeddings & Store in Vector Database
        print("Step 7 & 8: Generating embeddings and storing in ChromaDB...")
        vector_db.add_texts(texts=chunks, metadatas=metadatas)
        
    # Step 9: Knowledge Base Creation
    print("\nStep 9: Knowledge Base Creation Complete! Data is now searchable.")

if __name__ == "__main__":
    # We will search for 1 real paper about "Multimodal AI" to test the pipeline
    ingest_arxiv_papers("multimodal large language models", max_results=1)