<div align="center">
	<img src="frontend/public/syntropy-logo.png" alt="Syntropy logo" width="200" />

  <h1>Syntropy — Agentic RAG Research Assistant</h1>

</div>

[![Live Demo](https://img.shields.io/badge/Demo-Live-success?style=for-the-badge)](https://agentic-rag-research-assistant-lb5yixibp.vercel.app/)
[![Backend API](https://img.shields.io/badge/API-Deployed-blue?style=for-the-badge)](https://agentic-rag-backend-jy8a.onrender.com/)

> An autonomous AI research assistant that dynamically retrieves knowledge from a vector database or performs real-time academic research by discovering papers with Semantic Scholar and processing open-access PDFs.

Built using **LangGraph agent workflows**, **Supabase pgvector**, **Groq Llama-3.1**, and **Jina embeddings**.

---

# Project Overview

Syntropy is a **production-grade Agentic Retrieval-Augmented Generation (RAG) system**.

Unlike traditional RAG pipelines that only query a static vector database, this system **autonomously decides when it needs to expand its knowledge** by discovering new research papers through Semantic Scholar and adding them to its knowledge base.

This allows the assistant to **continuously grow its research knowledge during conversations.**

---

# Core Features

✅ **Agentic AI Workflow** powered by LangGraph  
✅ **Semantic Scholar powered paper discovery**  
✅ **Dynamic academic paper ingestion** through open-access PDFs  
✅ **Persistent Vector Database** using Supabase pgvector  
✅ **Fast LLM Inference** with Groq Llama-3.1  
✅ **Batch Embedding Pipeline** using Jina AI  
✅ **Conversational Memory** using Redis  
✅ **Context-aware follow-up question retrieval**  
✅ **Knowledge-base deduplication**  
✅ **Retrieval-aware research triggering**  
✅ **Real-time Research Retrieval**  
✅ **Production Deployment** on Render + Vercel

---

# System Architecture

![System Architecture](system_architecture.png)

---

# Tech Stack

## Backend

| Technology             | Purpose                              |
| ---------------------- | ------------------------------------ |
| FastAPI                | High-performance Python API          |
| LangGraph              | Agent workflow orchestration         |
| LangChain              | LLM abstraction + prompt handling    |
| Supabase pgvector      | Persistent vector database           |
| Groq Cloud             | Llama-3.1 inference                  |
| Jina AI                | High-speed embeddings API            |
| PyMuPDF                | PDF text extraction                  |
| Redis                  | Conversation memory persistence      |
| Semantic Scholar API   | Academic paper discovery and ranking |
| ArXiv/Open Access PDFs | Paper download source                |

---

## Frontend

| Technology  | Purpose            |
| ----------- | ------------------ |
| Next.js     | React framework    |
| TypeScript  | Type-safe frontend |
| TailwindCSS | UI styling         |
| Vercel      | Frontend hosting   |

---

## DevOps

| Tool     | Purpose                 |
| -------- | ----------------------- |
| Render   | Backend deployment      |
| Vercel   | Frontend deployment     |
| GitHub   | Version control         |
| Supabase | Vector database hosting |

---

# Deployment Architecture

```

User
↓
Vercel (Next.js Frontend)
↓
Render (FastAPI Backend)
↓
LangGraph Agent
↓
Supabase pgvector
↓
Jina Embedding API
↓
Groq Llama-3.1

```

---

# Live Deployment

Frontend  
https://agentic-rag-research-assistant-lb5yixibp.vercel.app

Backend API  
https://agentic-rag-backend-jy8a.onrender.com

---

# Key Technical Highlights

## 1️⃣ Agentic Decision Engine

The system uses **LangGraph's state machine** to determine whether existing knowledge is sufficient or new research must be performed.

```

retrieve_and_check
↓
decision
├─ generate_answer
└─ do_research

```

If the retrieved context is sufficient, the agent answers directly. If not, it automatically triggers the research pipeline.

---

# 2️⃣ Research Paper Ingestion Pipeline

When the system detects missing knowledge:

Before ingestion, existing papers are checked so duplicates are not re-added to Supabase.

1️⃣ User query  
2️⃣ LLM-generated research query  
3️⃣ Semantic Scholar search  
4️⃣ Paper metadata retrieval  
5️⃣ Direct PDF download from ArXiv or open-access URLs  
6️⃣ Extract text using PyMuPDF  
7️⃣ Split text into semantic chunks  
8️⃣ Generate embeddings using Jina API  
9️⃣ Store vectors in Supabase pgvector  
1️⃣0️⃣ Retrieve relevant chunks for answer generation

---

# 3️⃣ Optimized Embedding Pipeline

The system uses **batch embeddings** to dramatically reduce latency.

Instead of:

```

100 chunks → 100 API calls

```

It performs:

```

100 chunks → ~3 batch calls

```

Benefits:

- significantly faster ingestion
- reduced API calls
- lower embedding latency

The current workflow also reduces external research calls by skipping ingestion when retrieval already returns sufficient relevant context.

---

# 4️⃣ Persistent Vector Database

Vectors are stored in **Supabase pgvector**, enabling:

• persistent storage  
• scalable vector search  
• SQL-based similarity queries

Example vector similarity function:

```

match_documents(query_embedding vector, match_count int)

```

---

# Conversational Memory

User conversations are stored in **Redis**.

This enables:

• multi-turn conversation context  
• follow-up questions  
• scalable memory for multiple users

Follow-up questions are rewritten with prior conversation context before retrieval so short prompts like "explain more" stay anchored to the active topic.

---

# Performance Metrics

| Metric          | Value      |
| --------------- | ---------- |
| LLM inference   | ~400-600ms |
| Vector search   | ~50-100ms  |
| Paper ingestion | ~3-8s      |
| Cold start      | ~2s        |

Recent workflow improvements include fewer external research calls, faster paper acquisition, removal of ArXiv API rate-limit bottlenecks, and better retrieval relevance from higher-quality paper discovery.

Retrieval results also log source names and similarity scores, which makes runtime debugging of search quality easier.

---

# Repository Structure

```

Agentic_RAG/
│
├── backend/
│   ├── main.py
│   ├── agent.py
│   ├── ingest.py
│   ├── requirements.txt
│   └── runtime.txt
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── next.config.ts
│   └── package.json
│
├── system_architecture.png
├── README.md
└── .gitignore

```

---

# Environment Configuration

## Backend (.env)

```

GROQ_API_KEY=
JINA_API_KEY=
SUPABASE_URL=
SUPABASE_KEY=
REDIS_URL=
SEMANTIC_SCHOLAR_API_KEY=

```

---

## Frontend (.env.local)

```

NEXT_PUBLIC_BACKEND_URL=

```

---

# Future Improvements

Planned upgrades:

- Streaming responses (Server-Sent Events)
- Multi-source research ingestion
- Semantic caching layer
- Paper summarization
- Authentication (Clerk / Auth0)
- Query analytics dashboard
- Source-diverse retrieval
- Retrieval reranking
- Citation-aware answer generation
- Hybrid search (vector + keyword)

---

# Author

**Harshal Sharma**

AI / ML Engineer | Full-Stack AI Systems

GitHub  
https://github.com/Harshalsharma05

LinkedIn  
https://www.linkedin.com/in/harshal-sharma-98851b2ab

---

# Acknowledgements

Groq  
LangChain  
LangGraph  
Supabase  
Jina AI  
ArXiv

---

⭐ If you find this project interesting, consider giving the repository a star.
