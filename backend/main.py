from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
import os

app = FastAPI(title="AI Research Copilot API")

origins = [
    "https://agentic-rag-research-assistant.vercel.app",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy import agents to avoid slow startup
_agent_app = None
def get_agent():
    global _agent_app
    if _agent_app is None:
        from agent import agent_app
        _agent_app = agent_app
    return _agent_app

@app.get("/")
def read_root():
    return {"message": "AI Research Copilot Backend is running!", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# 1. Update Request to expect a session_id instead of the full chat_history array
class QueryRequest(BaseModel):
    query: str
    session_id: str 

@app.post("/api/chat")
def chat_with_llm(request: QueryRequest):
    print(f"\n[NEW REQUEST] {request.query} | [SESSION] {request.session_id}")

    formatted_history = []
    redis_history = None

    # Redis-backed history is optional. If Redis is unavailable, continue stateless.
    try:
        from langchain_community.chat_message_histories import RedisChatMessageHistory

        redis_url = os.environ.get("REDIS_URL")
        redis_history = RedisChatMessageHistory(
            session_id=request.session_id,
            url=redis_url or "redis://localhost:6379"
        )

        for msg in redis_history.messages:
            if msg.type == "human":
                role = "user"
            elif msg.type == "ai":
                role = "assistant"
            else:
                continue

            formatted_history.append({
                "role": role,
                "content": msg.content
            })

        print(f"[REDIS MEMORY] Loaded {len(formatted_history)} previous messages.")
    except Exception as e:
        print(f"[REDIS WARNING] {str(e)}")
        print("[REDIS WARNING] Continuing without persisted chat history.")

    try:
        agent = get_agent()
        initial_state = {
            "query": request.query,
            "chat_history": formatted_history,
            "context": "",
            "sources": [],
            "needs_research": False,
            "response": ""
        }

        print("[STATE SENT TO AGENT]", initial_state)
        result = agent.invoke(initial_state)
    except Exception as e:
        print("\n===== AGENT FULL ERROR =====")
        traceback.print_exc()
        print("============================\n")

        raise HTTPException(
            status_code=500,
            detail="Agent processing failed. Check backend logs."
        )

    if redis_history is not None:
        try:
            redis_history.add_user_message(request.query)
            redis_history.add_ai_message(result["response"])
        except Exception as e:
            print(f"[REDIS SAVE WARNING] {str(e)}")

    return {
        "response": result.get("response", ""),
        "sources": result.get("sources", [])
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use Render's PORT or default to 8000 for local
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=300)