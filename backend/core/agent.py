import os
import re
import json
import requests
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any
from pydantic.v1 import SecretStr

from supabase import create_client
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.services.utils import sanitize_search_query, build_research_queries, format_history, is_follow_up_query, build_contextual_query
from backend.services.embeddings import get_embedding

load_dotenv()

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

# LLM (Groq)

# 1. Groq - Keep this for your extremely fast workflow routing checks
routing_llm = ChatGroq(
    temperature=0,
    model="llama-3.1-8b-instant",
    api_key=SecretStr(require_env("GROQ_API_KEY")),
    stop_sequences=[]
)

# 2. Gemini - Use this for generation (massive context and structural reasoning)
generation_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    api_key=SecretStr(require_env("GEMINI_API_KEY")),
    temperature=0
)

# Supabase Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(require_env("SUPABASE_URL"), require_env("SUPABASE_KEY"))

# Jina Embedding API

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_RERANKER_MODEL = os.getenv("JINA_RERANKER_MODEL", "jina-reranker-v2-base-multilingual")

session = requests.Session()


# Agent State

class AgentState(TypedDict): # This is the state that will be passed through each node of the agent's workflow. this is for langgraph's StateGraph and allows us to have a clear and consistent structure for the data that the agent operates on. By defining this as a TypedDict, we get type checking and autocompletion benefits when working with the agent's state in our code.
    query: str
    chat_history: List[Dict[str, Any]]
    context: str
    sources: List[str]
    needs_research: bool
    response: str


# Supabase Retrieval

def retrieve_documents(query: str):

    query_embedding = get_embedding(query, task="retrieval.query")

    # Call your new hybrid search function passing both text and vectors
    response = supabase.rpc(
        "hybrid_search",
        {
            "query_text": query,
            "query_embedding": query_embedding,
            "match_count": 5
        }
    ).execute()

    response_rows = response.data if isinstance(response.data, list) else []

    if not response_rows:
        return [], []

    # Extract structural components from your candidate matches
    raw_docs = [str(row["content"]) for row in response_rows if isinstance(row, dict) and "content" in row]
    if not raw_docs:
        return [], []

    print(f"[HYBRID SEARCH] Fetched {len(raw_docs)} candidate chunks. Sending to Jina Reranker...")

    # 1. POST to Jina's Cross-Encoder endpoint
    rerank_response = session.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": JINA_RERANKER_MODEL,
            "query": query,
            "documents": raw_docs,
            "top_n": 4 # Dynamically drop the lowest scoring chunk to keep context window hyper-focused
        },
        timeout=30
    )

    if rerank_response.status_code != 200:
        print(f"[RERANKER WARNING] API failed with code {rerank_response.status_code}. Falling back to hybrid scores.")
        sorted_rows = response_rows
    else:
        # 2. Re-order your documents based on the precise contextual relevance score
        rerank_data = rerank_response.json().get("results", [])
        sorted_rows = []
        for item in rerank_data:
            idx = item["index"]
            raw_row = response_rows[idx]
            
            # Ensure it's treated strictly as a dictionary to appease Pylance
            if isinstance(raw_row, dict):
                row_dict = dict(raw_row)  # Create a mutable copy
                row_dict["rerank_score"] = item["relevance_score"]
                sorted_rows.append(row_dict)

    # 3. Parse final structured arrays making sure Pylance is happy with the types
    documents = [str(row["content"]) for row in sorted_rows if isinstance(row, dict) and "content" in row]
    
    metadatas = []
    for row in sorted_rows:
        if isinstance(row, dict):
            m = row.get("metadata", {})
            if isinstance(m, str):
                try: m = json.loads(m)
                except Exception: m = {}
                m["rerank_score"] = row.get("rerank_score", 0.0)
            
            metadatas.append(m if isinstance(m, dict) else {})
        else:
            metadatas.append({})

    # Log your precision metrics to verify that the out-of-context blocks were pushed down
    sources_with_scores = [
        f"{meta.get('source', 'Unknown')} (Rerank Score: {row.get('rerank_score', 0):.4f})"
        for meta, row in zip(metadatas, sorted_rows)
        if isinstance(row, dict) # Ensure row is a dict before accessing keys
    ]
    print(f"[FINAL RETRIEVAL CONTEXT] {len(documents)} chunks selected. Sources: {sources_with_scores}")

    return documents, metadatas


# Node 1: Retrieve Knowledge

def retrieve_and_check(state: AgentState):

    print("--- NODE: RETRIEVE & CHECK ---")

    query = state["query"]
    history_str = format_history(state["chat_history"])
    contextual_query = build_contextual_query(query, state["chat_history"])

    if contextual_query != query:
        print(f"[QUERY REWRITE] Retrieval query expanded to: {contextual_query}")

    documents, metadatas = retrieve_documents(contextual_query)
    
    # Jina Reranker v2 relevance scores range from 0.0 to 1.0. 
    # Anything below 0.35 is complete noise. We scrub out low-confidence matches instantly.
    filtered_pairs = [
        (doc, meta) for doc, meta in zip(documents, metadatas)
        if isinstance(meta, dict) and meta.get("rerank_score", 0.0) >= 0.35
    ]
    
    if filtered_pairs:
        documents = [pair[0] for pair in filtered_pairs]
        metadatas = [pair[1] for pair in filtered_pairs]
    else:
        documents, metadatas = [], []
        print("[RERANKER SHIELD] Wiped out all chunks. Confidence was below minimum threshold (0.35).")

    context_text = "\n\n".join(str(document) for document in documents)

    sources = list(
        {
            metadata.get("source", "Unknown")
            for metadata in metadatas
            if metadata
        }
    )
    sources = [s for s in sources if s != "Unknown"]

    # Fast path: if retrieval already found enough chunks, don't waste an LLM call deciding.

    needs_research = False

    # If the DB returned nothing at all, immediately trigger research
    if not documents:
        needs_research = True
        print("Knowledge Check: NO (No documents found)")
    else:
        # ALWAYS let Groq evaluate if the retrieved nearest-neighbors are actually relevant
        decision_prompt = f"""
            You are a strict RAG Retrieval Quality Evaluator.
            Your job is to determine if the current Context contains enough specific facts to directly answer or meaningfully contribute to the User's Query.

            Evaluation Rules:
            1. Reply YES if the context contains direct answers, partial answers, or highly relevant technical context that allows a grounded response.
            2. Reply NO if the context is completely empty, entirely unrelated to the core technical concepts, or if it is impossible to answer the query without making things up.
            3. Do NOT trigger unnecessary research if the context already contains the specific formulas, definitions, or mechanisms requested.

            Strict Output Format:
            Reply with exactly ONE word: either YES or NO. Do not include punctuation, reasoning, or extra characters.

            Conversation History:
            {history_str}

            User Query:
            {contextual_query}

            Retrieved Context:
            {context_text}

            Decision (YES/NO):
        """
        
        decision_message = routing_llm.invoke(decision_prompt)
        decision = str(getattr(decision_message, "content", "")).strip().upper()

        # If Groq says NO (the chunks are irrelevant), trigger research
        needs_research = "NO" in decision
        print("Knowledge Check:", decision)

    return {
        "query": state["query"],
        "chat_history": state["chat_history"],
        "context": context_text,
        "sources": sources,
        "needs_research": needs_research, # how to route the workflow based on whether we need to do research or can generate an answer directly from retrieved knowledge is determined in the routing function below, which checks this flag in the state to decide the next node in the workflow.
        "response": state.get("response", "")
    }


# Node 2: Research

def do_research(state: AgentState): # This node is responsible for performing research when the agent determines that the current knowledge base does not contain sufficient information to answer the user's query. It uses the LLM to generate a search query for ArXiv, ingests new papers based on that query, and then retrieves updated context from the knowledge base to be used in the final answer generation step.

    print("--- NODE: DO RESEARCH ---")

    query = state["query"]
    history_str = format_history(state["chat_history"])
    contextual_query = build_contextual_query(query, state["chat_history"])

    if contextual_query != query:
        print(f"[QUERY REWRITE] Research query expanded to: {contextual_query}")

    search_prompt = f"""
        You are a highly precise academic router for Semantic Scholar.
        Analyze the user's input and generate an optimized plain-text query.

        Strict Rules:
        1. If the user is asking about a specific, famous, or named paper (e.g., "Attention Is All You Need", "YOLOv8", "BERT"), output ONLY the exact, clean title of that paper. Do NOT add concepts, topics, or technical parameters onto it.
        2. If the user is asking a general topical question (e.g., "how does multi-head attention work?"), output 2 to 4 dense, specialized keyword noun phrases.
        3. Never include conversational verbs, formatting, labels, or stop words (e.g., "explain", "paper", "review", "concept").
        4. Return ONLY the plain search string. No quotes, no preamble.

        Examples:
        - Question: "Explain self-attention in Attention Is All You Need" -> Attention Is All You Need
        - Question: "What is the architecture of the original Transformer paper?" -> Attention Is All You Need
        - Question: "Show me recent optimizations for flash attention layers" -> flash attention optimization training

        Conversation History:
        {history_str}

        Question:
        {contextual_query}
    """

    raw_arxiv_message = routing_llm.invoke(search_prompt)
    raw_arxiv_query = str(getattr(raw_arxiv_message, "content", "")).strip()
    arxiv_queries = build_research_queries(contextual_query, raw_arxiv_query)
    research_query = arxiv_queries[0] if arxiv_queries else sanitize_search_query(contextual_query)

    print("Research Raw:", raw_arxiv_query)
    print("Research Search Sanitized:", research_query)

    arxiv_sources = []
    try:
        from backend.services.ingest import ingest_arxiv_papers # ingest is the function we defined in ingest.py that searches ArXiv based on the provided query, downloads the papers, extracts text, and ingests it into Supabase. It returns a list of source labels for the ingested papers, which we can use to provide source information to the user in the final response.
        ingested = ingest_arxiv_papers(research_query, max_results=10, max_papers_to_ingest=2)
        if ingested:
            arxiv_sources = ingested
    except Exception as e:
        print("[RESEARCH WARNING]", str(e))

    documents, metadatas = retrieve_documents(contextual_query)
    
    filtered_pairs = [
        (doc, meta) for doc, meta in zip(documents, metadatas)
        if isinstance(meta, dict) and meta.get("rerank_score", 0.0) >= 0.35
    ]
    
    if filtered_pairs:
        documents = [pair[0] for pair in filtered_pairs]
        metadatas = [pair[1] for pair in filtered_pairs]
    else:
        documents, metadatas = [], []
        print("[RERANKER SHIELD - RESEARCH] Wiped out low-confidence fallback context.")

    context_text = "\n\n".join(str(document) for document in documents)

    sources = list(
        {
            metadata.get("source", "Unknown")
            for metadata in metadatas
            if metadata
        }
    )

    # Remove uninformative "Unknown" entries; fall back to arxiv paper titles
    sources = [s for s in sources if s != "Unknown"]
    if not sources:
        sources = arxiv_sources

    return {
        "query": state["query"],
        "chat_history": state["chat_history"],
        "context": context_text,
        "sources": sources,
        "needs_research": False,
        "response": state.get("response", "")
    }


# Node 3: Generate Answer

def generate_answer(state: AgentState):

    print("--- NODE: GENERATE ANSWER ---")

    context_text = state["context"]
    query = state["query"]
    history_str = format_history(state["chat_history"])
    sources_text = "\n".join(f"- {source}" for source in state["sources"]) if state["sources"] else "- No sources available"

    if not context_text.strip():
        return {
            "query": state["query"],
            "chat_history": state["chat_history"],
            "context": state["context"],
            "sources": state["sources"],
            "needs_research": False,
            "response": "I could not find enough information in the retrieved papers to answer this question.",
        }

    prompt = f"""
        You are an AI Research Assistant.

        Conversation History:
        {history_str}

        Use only the retrieved context below as evidence.
        Do not use outside knowledge, prior training, or conversation history as factual evidence.
        If the answer is not explicitly supported by the context, say you could not find it in the retrieved papers.
        Do not speculate.
        Keep the answer concise and grounded.

        Retrieved Sources:
        {sources_text}

        Context:
        {context_text}

        Question:
        {query}

        Answer using only the retrieved context. If needed, begin with: "I could not find enough information in the retrieved papers."
    """

    response_message = generation_llm.invoke(prompt)
    # response = str(getattr(response_message, "content", "")).strip()
    response = str(response_message.content).strip()

    return {
        "query": state["query"],
        "chat_history": state["chat_history"],
        "context": state["context"],
        "sources": state["sources"],
        "needs_research": False,
        "response": response
    }


# Routing Logic

def route_research(state: AgentState): # It checks the "needs_research" flag in the state to determine whether the agent should proceed to the "do_research" node (if more research is needed) or skip directly to the "generate_answer" node (if the retrieved knowledge is sufficient). This allows the agent to dynamically adjust its workflow based on the information it has and the user's query.

    if state["needs_research"]:
        return "do_research"

    return "generate_answer"


# Build Graph

workflow = StateGraph(AgentState) # This initializes a new StateGraph from the langgraph library, which allows us to define a workflow of nodes (functions) that operate on a shared state (AgentState). Each node can modify the state and pass it to the next node in the workflow. The graph also supports conditional routing based on the state, which we use to determine whether to perform research or generate an answer directly based on the retrieved knowledge.

workflow.add_node("retrieve_and_check", retrieve_and_check)
workflow.add_node("do_research", do_research)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("retrieve_and_check")

workflow.add_conditional_edges( # This adds conditional routing logic to the workflow. After the "retrieve_and_check" node is executed, the "route_research" function is called with the current state. This function checks the "needs_research" flag in the state to determine whether the agent should proceed to the "do_research" node (if more research is needed) or skip directly to the "generate_answer" node (if the retrieved knowledge is sufficient). This allows the agent to dynamically adjust its workflow based on the information it has and the user's query.
    "retrieve_and_check",
    route_research
)

workflow.add_edge("do_research", "generate_answer") # After the "do_research" node is executed, we always want to proceed to the "generate_answer" node to produce the final response for the user, using the updated context and sources obtained from the research step.

workflow.add_edge("generate_answer", END) # END is a special marker from langgraph that indicates the end of the workflow. After the "generate_answer" node is executed, the workflow will terminate and return the final state, which includes the generated response and any relevant sources, back to the caller (in this case, our FastAPI endpoint in main.py).

agent_app = workflow.compile() # This compiles the defined workflow into an executable agent application that can be invoked with an initial state. This allows us to call agent_app.invoke(initial_state) to run the entire workflow synchronously, passing the state through each node and applying the defined logic and routing until we reach the end of the workflow, at which point we get the final state with the generated response ready to be returned to the user.