from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from langchain_community.chat_message_histories import RedisChatMessageHistory
from agent import agent_app # Import our LangGraph Agent!
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="AI Research Copilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allowing all origins for ease of local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Update Request to expect a session_id instead of the full chat_history array
class QueryRequest(BaseModel):
    query: str
    session_id: str 

@app.post("/api/chat")
def chat_with_llm(request: QueryRequest):
    print(f"\n[NEW REQUEST] {request.query} | [SESSION] {request.session_id}")
    
    # 2. Connect to Memurai (Redis) on default port 6379
    redis_url = os.environ.get("REDIS_URL")
    redis_history = RedisChatMessageHistory(
        request.session_id, 
        url=redis_url or "redis://localhost:6379"
    )
    
    # 3. Format Redis history into the dict format our agent.py expects
    formatted_history =[]
    for msg in redis_history.messages:
        role = "user" if msg.type == "human" else "assistant"
        formatted_history.append({"role": role, "content": msg.content})
        
    print(f"[REDIS MEMORY] Loaded {len(formatted_history)} previous messages.")
    
    # 4. Run the Agent
    initial_state = {
        "query": request.query, 
        "chat_history": formatted_history,
        "context": "", 
        "sources":[], 
        "needs_research": False, 
        "response": ""
    }
    
    result = agent_app.invoke(initial_state)
    
    # 5. Save the new conversation turn back to Redis!
    redis_history.add_user_message(request.query)
    redis_history.add_ai_message(result["response"])
    
    return {
        "response": result["response"],
        "sources": result["sources"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use Render's PORT or default to 8000 for local
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=300)