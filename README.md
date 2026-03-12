# 🤖 AI Research Copilot - Agentic RAG System

> **An autonomous AI research assistant** that intelligently retrieves information from a vector database or dynamically fetches and processes ArXiv papers on-demand using LangGraph decision workflows.

[![Live Demo](https://img.shields.io/badge/Demo-Live-success?style=for-the-badge)](https://your-vercel-url.vercel.app)
[![Backend API](https://img.shields.io/badge/API-Deployed-blue?style=for-the-badge)](https://your-render-url.onrender.com)

---

## 🎯 Project Overview

AI Research Copilot is a production-grade **Retrieval-Augmented Generation (RAG)** system enhanced with **agentic workflows** powered by LangGraph. Unlike traditional RAG systems, this agent autonomously decides whether to answer from existing knowledge or trigger real-time research paper ingestion from ArXiv.

### 🔑 Key Features

✅ **Autonomous Agent Architecture** - LangGraph-based decision engine  
✅ **Dynamic Knowledge Expansion** - Auto-downloads & processes research papers  
✅ **Conversational Memory** - Multi-turn dialogue with context retention (3 follow-ups)  
✅ **Real-time Inference** - Sub-second responses via Groq Llama 3.1  
✅ **Source Attribution** - Automatic citation generation  
✅ **Production Deployment** - Backend on Render, Frontend on Vercel

---

## 🏗️ System Architecture

![System Architecture](system_architecture.png)

### Workflow Pipeline

1. **User Query** → Next.js Frontend
2. **API Gateway** → FastAPI Backend
3. **Agent Controller** (LangGraph) evaluates:
   - **Path A**: Retrieve from ChromaDB if knowledge exists
   - **Path B**: Trigger ArXiv search → Download PDF → Extract text → Chunk → Embed → Store → Retrieve
4. **LLM Generation** → Groq Llama 3.1 synthesizes response with citations
5. **Response** → Frontend with sources

---

## 💻 Tech Stack

### Backend

| Technology                | Purpose                                         |
| ------------------------- | ----------------------------------------------- |
| **FastAPI**               | High-performance async REST API                 |
| **LangGraph**             | Agentic workflow orchestration & decision logic |
| **LangChain**             | LLM framework & prompt management               |
| **ChromaDB**              | Vector database for semantic search             |
| **Groq Cloud**            | Llama 3.1 inference (100+ tokens/sec)           |
| **ArXiv API**             | Research paper retrieval                        |
| **PyMuPDF**               | PDF text extraction                             |
| **Sentence Transformers** | Text embeddings (all-MiniLM-L6-v2)              |

### Frontend

| Technology      | Purpose                                    |
| --------------- | ------------------------------------------ |
| **Next.js 16**  | React framework with server-side rendering |
| **TypeScript**  | Type-safe development                      |
| **TailwindCSS** | Responsive UI styling                      |
| **API Proxy**   | CORS-free backend communication            |

### DevOps

| Tool           | Purpose                          |
| -------------- | -------------------------------- |
| **Render**     | Backend deployment (FastAPI)     |
| **Vercel**     | Frontend deployment (Next.js)    |
| **Git/GitHub** | Version control & CI/CD triggers |

---

## 🚀 Deployment Architecture

```
User Request
    ↓
Vercel (Next.js Frontend)
    ↓ HTTPS
Render (FastAPI Backend)
    ↓
LangGraph Agent
    ├─→ ChromaDB (Vector Search)
    ├─→ ArXiv API (Paper Fetch)
    └─→ Groq Cloud (LLM Inference)
```

**Live URLs:**

- Frontend: `https://your-vercel-url.vercel.app`
- Backend API: `https://your-render-url.onrender.com`

---

## 🧠 Technical Highlights

### 1. **Agentic Decision Making**

The system uses LangGraph's state machine to autonomously decide whether to:

- Answer from existing vector DB knowledge
- Trigger real-time paper ingestion pipeline

```python
# Simplified agent logic
workflow.add_conditional_edges(
    "retrieve_and_check",
    route_research  # Autonomous routing based on knowledge availability
)
```

### 2. **RAG Pipeline**

- **Document Processing**: PyMuPDF extracts text from ArXiv PDFs
- **Chunking**: RecursiveCharacterTextSplitter (1000 chars, 100 overlap)
- **Embedding**: Sentence Transformers (all-MiniLM-L6-v2, 384-dim vectors)
- **Storage**: ChromaDB with metadata for citation tracking
- **Retrieval**: Top-K semantic similarity search (k=3)

### 3. **Conversational Context**

- Session-based chat history management
- Context window: 3 follow-up questions per topic
- Redis-backed message history (scalable for multi-user)

### 4. **Production-Ready Features**

- CORS configuration for cross-origin requests
- Environment-based configuration (.env management)
- Error handling with detailed HTTP status codes
- Timeout handling for long-running LLM requests
- Animated loading states for better UX

---

## 📊 Performance Metrics

| Metric               | Value                              |
| -------------------- | ---------------------------------- |
| **LLM Inference**    | ~500ms (Groq Llama 3.1)            |
| **Vector Search**    | <100ms (ChromaDB)                  |
| **PDF Ingestion**    | ~10s per paper (ArXiv → Vector DB) |
| **Total Cold Start** | <2s (without paper download)       |

---

## 📁 Repository Structure

```
Agentic_RAG/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── agent.py             # LangGraph agentic workflow
│   ├── ingest.py            # ArXiv paper ingestion pipeline
│   ├── requirements.txt     # Python dependencies
│   └── render.yaml          # Render deployment config
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx         # Chat interface with real-time updates
│   │   ├── layout.tsx       # App layout & metadata
│   │   └── globals.css      # TailwindCSS styles
│   ├── next.config.ts       # API proxy & build config
│   └── package.json         # Node.js dependencies
│
├── system_architecture.png  # Architecture diagram
├── README.md               # This file
└── .gitignore              # Exclude secrets & build artifacts
```

---

## 🔐 Environment Configuration

### Backend (.env)

```env
GROQ_API_KEY=<your_groq_api_key>
```

### Frontend (.env.local)

```env
NEXT_PUBLIC_BACKEND_URL=<your_render_backend_url>
```

---

## 🌟 Future Enhancements

- [ ] Multi-user authentication (Auth0/Clerk)
- [ ] Persistent vector DB (Pinecone/Weaviate)
- [ ] Streaming responses (Server-Sent Events)
- [ ] Advanced citation formatting (APA/MLA)
- [ ] Multi-source ingestion (Google Scholar, PubMed)
- [ ] Query analytics dashboard

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details

---

## 👤 Author

**[Your Name]**  
Data Science & Machine Learning | Full-Stack AI Engineer  
📧 your.email@example.com | 🔗 [LinkedIn](https://linkedin.com/in/yourprofile) | 💻 [GitHub](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- **Groq** for lightning-fast LLM inference
- **LangChain** for the powerful agent framework
- **ArXiv** for open research paper access
- **Render** & **Vercel** for seamless deployment

---

**⭐ If you find this project interesting, please consider giving it a star!**
