from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
import os
from collections import defaultdict, deque
from threading import Lock
import urllib.parse

import redis as redis_lib
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

app = FastAPI(title="AI Research Copilot API")

_session_history = defaultdict(lambda: deque(maxlen=20))
_session_history_lock = Lock()

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

# Lazy import agents to avoid slow startup due to heavy dependencies
_agent_app = None
def get_agent(): # this function will import the agent_app from agent.py on first call, and return the same instance on subsequent calls. This allows us to avoid the slow import of langchain and other heavy libraries until we actually need to process a request, improving startup time and resource usage for simple health checks or other non-agent endpoints.
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
    redis_url = os.environ.get("REDIS_URL")

    print(f"[REDIS CONFIG] REDIS_URL set: {'yes' if redis_url else 'no'}")
    if redis_url:
        parsed_url = urllib.parse.urlparse(redis_url)
        print(f"[REDIS CONFIG] Host: {parsed_url.hostname or 'unknown'} | Port: {parsed_url.port or 'default'} | Scheme: {parsed_url.scheme or 'unknown'}")

        try:
            redis_client = redis_lib.Redis.from_url(redis_url)
            ping_result = redis_client.ping()
            print(f"[REDIS PING] success={ping_result}")
        except Exception as ping_error:
            print(f"[REDIS PING WARNING] {str(ping_error)}")

    # Redis-backed history is optional. If Redis is unavailable, continue stateless.
    try:
        from langchain_community.chat_message_histories import RedisChatMessageHistory # this is a thin wrapper around Redis that formats messages in the way our agent expects, and automatically creates/loads history based on session_id (this is provided by langchain-redis, not custom code)

        redis_history = RedisChatMessageHistory( # this will create a new Redis entry if session_id doesn't exist, or load existing history if it does
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

        print(f"[REDIS MEMORY] Loaded {len(formatted_history)} previous messages from Redis.")
    except Exception as e:
        print(f"[REDIS WARNING] {str(e)}")

    with _session_history_lock:
        local_history = list(_session_history[request.session_id])

    if local_history:
        print(f"[SESSION MEMORY] Loaded {len(local_history)} local fallback messages.")

    if local_history:
        existing_pairs = {(msg["role"], msg["content"]) for msg in formatted_history}
        for message in local_history:
            message_pair = (message.get("role"), message.get("content"))
            if message_pair not in existing_pairs:
                formatted_history.append(message)
                existing_pairs.add(message_pair)

    if not formatted_history:
        print("[HISTORY WARNING] Continuing without persisted chat history.")
    else:
        print(f"[HISTORY READY] Using {len(formatted_history)} messages for this request.")

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
        result = agent.invoke(initial_state) # invoke method is provided by langchain's Agent class, and runs the agent synchronously, returning the final state after all steps are complete. This is a blocking call, but allows us to keep the agent logic simple and linear without needing to manage async or callbacks.
    except Exception as e:
        print("\n AGENT FULL ERROR")
        traceback.print_exc()
        print("============================\n")

        raise HTTPException(
            status_code=500,
            detail="Agent processing failed. Check backend logs."
        )

    if redis_history is not None:
        try:
            # add_user_message and add_ai_message are methods provided by RedisChatMessageHistory to append messages to the history in Redis. This ensures that the conversation history is persisted across requests and can be retrieved in future interactions using the same session_id.
            redis_history.add_user_message(request.query)
            redis_history.add_ai_message(result["response"])
        except Exception as e:
            print(f"[REDIS SAVE WARNING] {str(e)}")

    with _session_history_lock:
        _session_history[request.session_id].append({"role": "user", "content": request.query})
        _session_history[request.session_id].append({"role": "assistant", "content": result.get("response", "")})

    return {
        "response": result.get("response", ""),
        "sources": result.get("sources", [])
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use Render's PORT or default to 8000 for local
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=300)