import os
import requests
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any

import chromadb
from chromadb.config import Settings
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# -------------------------------
# LLM (Groq)
# -------------------------------

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)


# -------------------------------
# Jina Embedding API
# -------------------------------

JINA_API_KEY = os.getenv("JINA_API_KEY")

session = requests.Session()

def get_embedding(text: str):

    response = session.post(
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
            f"Jina embedding API error: {response.status_code} - {response.text}"
        )

    data = response.json()

    return data["data"][0]["embedding"]


class JinaEmbeddingFunction:

    def embed_documents(self, texts: List[str]):
        return [get_embedding(text) for text in texts]

    def embed_query(self, text: str):
        return get_embedding(text)


embeddings = JinaEmbeddingFunction()

# -------------------------------
# Vector Database
# -------------------------------

persist_directory = "./chroma_db"

chroma_client = chromadb.PersistentClient(
    path=persist_directory,
    settings=Settings(anonymized_telemetry=False)
)

vector_db = chroma_client.get_or_create_collection("langchain")

# -------------------------------
# Agent State
# -------------------------------

class AgentState(TypedDict):
    query: str
    chat_history: List[Dict[str, Any]]
    context: str
    sources: List[str]
    needs_research: bool
    response: str


# -------------------------------
# Helper: format conversation
# -------------------------------

def format_history(history: List[Dict[str, Any]]) -> str:

    if not history:
        return "No previous conversation."

    formatted_msgs = []

    for msg in history:
        role = "User" if msg["role"] == "user" else "AI Assistant"
        formatted_msgs.append(f"{role}: {msg['content']}")

    return "\n".join(formatted_msgs)


# -------------------------------
# Node 1: Retrieve Knowledge
# -------------------------------

def retrieve_and_check(state: AgentState):

    print("--- NODE: RETRIEVE & CHECK ---")

    query = state["query"]
    history_str = format_history(state["chat_history"])

    results = vector_db.query(
        query_embeddings=[embeddings.embed_query(query)],
        n_results=3,
    )

    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    documents = documents[0] if documents else []
    metadatas = metadatas[0] if metadatas else []

    context_text = "\n\n".join(documents)
    sources = list(
        {
            metadata.get("source", "Unknown")
            for metadata in metadatas
            if metadata
        }
    )

    decision_prompt = f"""
You are a strict evaluator. Read the conversation history, the user's latest query, and the retrieved context.

Does the retrieved context contain enough relevant information to answer the latest query?

Reply with EXACTLY "YES" or "NO".

Conversation History:
{history_str}

Latest Query:
{query}

Retrieved Context:
{context_text}
"""

    decision = llm.invoke(decision_prompt).content.strip().upper()

    print("Agent Knowledge Check Decision:", decision)

    needs_research = "NO" in decision or not context_text.strip()

    return {
        "query": state["query"],
        "chat_history": state["chat_history"],
        "context": context_text,
        "sources": sources,
        "needs_research": needs_research,
        "response": state.get("response", "")
    }


# -------------------------------
# Node 2: Research (Stage 2)
# -------------------------------

def do_research(state: AgentState):

    print("--- NODE: DO RESEARCH ---")

    query = state["query"]
    history_str = format_history(state["chat_history"])

    search_query_prompt = f"""
Based on the conversation history and the latest question,
generate a short 2–3 keyword search phrase for ArXiv.

ONLY output the keywords.

Conversation History:
{history_str}

Latest Question:
{query}
"""

    arxiv_query = llm.invoke(search_query_prompt).content.strip()

    print("Generated ArXiv Query:", arxiv_query)

    try:
        from ingest import ingest_arxiv_papers

        ingest_arxiv_papers(arxiv_query, max_results=1)
    except Exception as e:
        print(f"[RESEARCH WARNING] Failed to ingest new papers: {str(e)}")

    results = vector_db.query(
        query_embeddings=[embeddings.embed_query(query)],
        n_results=3,
    )

    documents = []
    metadatas = []

    if results and "documents" in results and results["documents"]:
        documents = results["documents"][0]

    if results and "metadatas" in results and results["metadatas"]:
        metadatas = results["metadatas"][0]

    if not documents:
        context_text = ""
    else:
        context_text = "\n\n".join(documents)
    
    sources = []
    for m in metadatas:
        if isinstance(m, dict) and "source" in m:
            sources.append(m["source"])
    sources = list(set(sources))    

    return {
        "query": state["query"],
        "chat_history": state["chat_history"],
        "context": context_text,
        "sources": sources,
        "needs_research": False,
        "response": state.get("response", "")
    }


# -------------------------------
# Node 3: Generate Answer
# -------------------------------

def generate_answer(state: AgentState):

    print("--- NODE: GENERATE ANSWER ---")

    context_text = state["context"]
    query = state["query"]
    history_str = format_history(state["chat_history"])

    prompt = f"""
You are an AI Research Assistant.

Use the retrieved research context and the conversation history to answer the user's latest question.

Conversation History:
{history_str}

Retrieved Context:
{context_text}

Latest Question:
{query}

Answer:
"""

    response = llm.invoke(prompt).content

    return {
        "query": state["query"],
        "chat_history": state["chat_history"],
        "context": state["context"],
        "sources": state["sources"],
        "needs_research": False,
        "response": response
    }


# -------------------------------
# Routing Logic
# -------------------------------

def route_research(state: AgentState):

    if state["needs_research"]:
        return "do_research"

    return "generate_answer"


# -------------------------------
# Build LangGraph Workflow
# -------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("retrieve_and_check", retrieve_and_check)
workflow.add_node("do_research", do_research)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("retrieve_and_check")

workflow.add_conditional_edges(
    "retrieve_and_check",
    route_research
)

workflow.add_edge("do_research", "generate_answer")
workflow.add_edge("generate_answer", END)

agent_app = workflow.compile()













































# Older Verison of agent.py using OLlama (llama3) on local computer
# from typing import TypedDict, List, Dict, Any
# from langgraph.graph import StateGraph, END
# from langchain_community.llms import Ollama
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from ingest import ingest_arxiv_papers

# # Initialize models and DB
# llm = Ollama(model="llama3")
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# persist_directory = "./chroma_db"
# vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# # 1. Define the Agent's "Memory" (State) - Added chat_history
# class AgentState(TypedDict):
#     query: str
#     chat_history: List[Dict[str, Any]]
#     context: str
#     sources: List[str]
#     needs_research: bool
#     response: str

# # Helper Function: Convert history dict to a readable string for the LLM
# def format_history(history: List[Dict[str, Any]]) -> str:
#     if not history:
#         return "No previous conversation."
    
#     formatted_msgs =[]
#     for msg in history:
#         role = "User" if msg["role"] == "user" else "AI Assistant"
#         formatted_msgs.append(f"{role}: {msg['content']}")
#     return "\n".join(formatted_msgs)

# # 2. Node 1: Retrieve and Check Knowledge
# def retrieve_and_check(state: AgentState):
#     print("--- NODE: RETRIEVE & CHECK ---")
#     query = state["query"]
#     history_str = format_history(state["chat_history"])
    
#     # Search existing DB
#     docs = vector_db.similarity_search(query, k=3)
#     context_text = "\n\n".join([doc.page_content for doc in docs])
#     sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
    
#     # Evaluator now knows the conversation history!
#     decision_prompt = f"""
#     You are a strict evaluator. Read the conversation history, the user's latest query, and the retrieved context.
#     Does the retrieved context contain enough relevant information to answer the latest query?
#     Reply with EXACTLY "YES" or "NO".
    
#     Conversation History:
#     {history_str}
    
#     Latest Query: {query}
#     Retrieved Context: {context_text}
#     """
#     decision = llm.invoke(decision_prompt).strip().upper()
#     print(f"Agent Knowledge Check Decision: {decision}")
    
#     needs_research = "NO" in decision or not context_text.strip()
    
#     return {"context": context_text, "sources": sources, "needs_research": needs_research}

# # 3. Node 2: Execute Research Tools (Stage 2 Pipeline)
# def do_research(state: AgentState):
#     print("--- NODE: DO RESEARCH (DOWNLOADING NEW PAPER) ---")
#     query = state["query"]
#     history_str = format_history(state["chat_history"])
    
#     # AI uses history to generate better search keywords for follow-ups
#     search_query_prompt = f"""
#     Based on the conversation history and the latest question, generate a 2 to 3 keyword search phrase for ArXiv.
#     ONLY output the keywords.
    
#     Conversation History:
#     {history_str}
    
#     Latest Question: {query}
#     """
#     arxiv_query = llm.invoke(search_query_prompt).strip()
#     print(f"Generated ArXiv Search Query: {arxiv_query}")
    
#     ingest_arxiv_papers(arxiv_query, max_results=1)
    
#     # Search the DB again now that we have new data
#     docs = vector_db.similarity_search(query, k=3)
#     context_text = "\n\n".join([doc.page_content for doc in docs])
#     sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
    
#     return {"context": context_text, "sources": sources}

# # 4. Node 3: LLM Reasoning (Generate Final Answer)
# def generate_answer(state: AgentState):
#     print("--- NODE: GENERATE ANSWER ---")
#     context_text = state["context"]
#     query = state["query"]
#     history_str = format_history(state["chat_history"])
    
#     # AI uses history to format its final response correctly
#     prompt = f"""You are an AI Research Assistant. Use the retrieved research context and the conversation history to answer the user's latest question. 
    
#     Conversation History:
#     {history_str}

#     Retrieved Context:
#     {context_text}

#     Latest Question:
#     {query}

#     Answer:"""
    
#     response = llm.invoke(prompt)
#     return {"response": response}

# # 5. Routing Logic & Graph Build
# def route_research(state: AgentState):
#     if state["needs_research"]:
#         return "do_research"
#     return "generate_answer"

# workflow = StateGraph(AgentState)
# workflow.add_node("retrieve_and_check", retrieve_and_check)
# workflow.add_node("do_research", do_research)
# workflow.add_node("generate_answer", generate_answer)

# workflow.set_entry_point("retrieve_and_check")
# workflow.add_conditional_edges("retrieve_and_check", route_research)
# workflow.add_edge("do_research", "generate_answer")
# workflow.add_edge("generate_answer", END)

# agent_app = workflow.compile()