import os
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

# Import Groq instead of Ollama
from langchain_groq import ChatGroq 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from ingest import ingest_arxiv_papers

# Load environment variables (API Key)
load_dotenv()

# Initialize Groq LLM (Using Llama 3.1 8B on Groq for blazing speed)
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.environ.get("GROQ_API_KEY")
)


embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
)

persist_directory = "./chroma_db"
vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# 1. Define the Agent's "Memory" (State)
class AgentState(TypedDict):
    query: str
    chat_history: List[Dict[str, Any]]
    context: str
    sources: List[str]
    needs_research: bool
    response: str

# Helper Function
def format_history(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "No previous conversation."
    
    formatted_msgs = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "AI Assistant"
        formatted_msgs.append(f"{role}: {msg['content']}")
    return "\n".join(formatted_msgs)

# 2. Node 1: Retrieve and Check Knowledge
def retrieve_and_check(state: AgentState):
    print("--- NODE: RETRIEVE & CHECK ---")
    query = state["query"]
    history_str = format_history(state["chat_history"])
    
    docs = vector_db.similarity_search(query, k=3)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
    
    decision_prompt = f"""
    You are a strict evaluator. Read the conversation history, the user's latest query, and the retrieved context.
    Does the retrieved context contain enough relevant information to answer the latest query?
    Reply with EXACTLY "YES" or "NO".
    
    Conversation History:
    {history_str}
    
    Latest Query: {query}
    Retrieved Context: {context_text}
    """
    
    # Notice the .content added here for Groq!
    decision = llm.invoke(decision_prompt).content.strip().upper()
    print(f"Agent Knowledge Check Decision: {decision}")
    
    needs_research = "NO" in decision or not context_text.strip()
    
    return {"context": context_text, "sources": sources, "needs_research": needs_research}

# 3. Node 2: Execute Research Tools (Stage 2 Pipeline)
def do_research(state: AgentState):
    print("--- NODE: DO RESEARCH (DOWNLOADING NEW PAPER) ---")
    query = state["query"]
    history_str = format_history(state["chat_history"])
    
    search_query_prompt = f"""
    Based on the conversation history and the latest question, generate a 2 to 3 keyword search phrase for ArXiv.
    ONLY output the keywords.
    
    Conversation History:
    {history_str}
    
    Latest Question: {query}
    """
    
    # Notice the .content added here for Groq!
    arxiv_query = llm.invoke(search_query_prompt).content.strip()
    print(f"Generated ArXiv Search Query: {arxiv_query}")
    
    ingest_arxiv_papers(arxiv_query, max_results=1)
    
    docs = vector_db.similarity_search(query, k=3)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
    
    return {"context": context_text, "sources": sources}

# 4. Node 3: LLM Reasoning (Generate Final Answer)
def generate_answer(state: AgentState):
    print("--- NODE: GENERATE ANSWER ---")
    context_text = state["context"]
    query = state["query"]
    history_str = format_history(state["chat_history"])
    
    prompt = f"""You are an AI Research Assistant. Use the retrieved research context and the conversation history to answer the user's latest question. 
    
    Conversation History:
    {history_str}

    Retrieved Context:
    {context_text}

    Latest Question:
    {query}

    Answer:"""
    
    # Notice the .content added here for Groq!
    response = llm.invoke(prompt).content
    return {"response": response}

# 5. Routing Logic & Graph Build
def route_research(state: AgentState):
    if state["needs_research"]:
        return "do_research"
    return "generate_answer"

workflow = StateGraph(AgentState)
workflow.add_node("retrieve_and_check", retrieve_and_check)
workflow.add_node("do_research", do_research)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("retrieve_and_check")
workflow.add_conditional_edges("retrieve_and_check", route_research)
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