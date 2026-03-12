# 🤖 AI Research Copilot - Agentic RAG System

An intelligent research assistant powered by **LangGraph** agentic workflows, **ChromaDB** vector storage, and **Groq's Llama 3.1** for blazing-fast inference. Features automatic ArXiv paper ingestion and conversational memory.

## 🏗️ Architecture

```
Frontend (Next.js + React)
    ↓ API Proxy
Backend (FastAPI)
    ↓
Agent (LangGraph)
    ├─ Retrieve & Check Knowledge
    ├─ Auto Research (ArXiv Ingestion)
    └─ Generate Answer (Groq Llama 3.1)
    ↓
Vector DB (ChromaDB + HuggingFace Embeddings)
```

## ✨ Features

- **Conversational Memory**: Multi-turn chat with context awareness (3 follow-ups)
- **Agentic Workflow**: Smart decision-making - retrieves existing knowledge or fetches new papers automatically
- **Real-time Paper Ingestion**: Downloads & processes ArXiv papers on-demand
- **Fast Inference**: Groq cloud for sub-second LLM responses
- **Source Citations**: All answers include paper references

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Groq API Key ([Get it here](https://console.groq.com/))

### 1️⃣ Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv ai-research
ai-research\Scripts\activate    # Windows
# source ai-research/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
# Create .env file with:
# GROQ_API_KEY=your_groq_api_key_here

# Run backend server
python main.py
```

Backend runs on: `http://localhost:8000`

### 2️⃣ Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend runs on: `http://localhost:3000`

## 📁 Project Structure

```
Agentic_RAG/
├── backend/
│   ├── main.py          # FastAPI server
│   ├── agent.py         # LangGraph agentic workflow
│   ├── ingest.py        # ArXiv paper ingestion
│   ├── requirements.txt # Python dependencies
│   └── .env            # API keys (not committed)
│
└── frontend/
    ├── app/
    │   ├── page.tsx    # Main chat interface
    │   └── globals.css # Styling
    └── next.config.ts  # API proxy configuration
```

## 🔧 Configuration

### Backend (`.env`)

```env
GROQ_API_KEY=gsk_your_api_key_here
```

### Frontend (`next.config.ts`)

API calls to `/api/chat` are proxied to `http://127.0.0.1:8000/api/chat`

## 🧪 Testing

```bash
# Test backend endpoint
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What is GPT-4V?","chat_history":[]}'
```

## 📚 Tech Stack

**Backend:**

- FastAPI - Web framework
- LangGraph - Agentic workflow orchestration
- LangChain - LLM framework
- ChromaDB - Vector database
- Groq - LLM inference (Llama 3.1)
- ArXiv API - Research paper retrieval
- PyMuPDF - PDF text extraction

**Frontend:**

- Next.js 16 - React framework
- TypeScript - Type safety
- TailwindCSS - Styling

## 🔒 Security Notes

⚠️ **Never commit your `.env` file!** It contains sensitive API keys.

## 📝 License

MIT

## 👤 Author

Data Science & ML Research Project - 2026
