# Codebase Intelligence

Ask questions about any public GitHub repository in plain English and get accurate, grounded answers with source citations — powered by Claude Sonnet + ChromaDB + RAG.

## What it does

1. You paste a GitHub repo URL and click **Index Repo**
2. The backend clones the repo, chunks all code files into 500-token segments, embeds them with `all-MiniLM-L6-v2`, and stores them in ChromaDB
3. You ask a question in the chat interface
4. The backend retrieves the top 15 relevant chunks via cosine similarity, reranks them with a cross-encoder, and sends the top 5 to Claude Sonnet
5. Claude streams a grounded answer back token-by-token, citing the exact files it used

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Next.js Frontend                      │
│  ┌──────────────┐   ┌──────────────────────────────┐   │
│  │  Index Panel │   │      Chat Interface           │   │
│  │  (repo URL)  │   │  (streaming answer + cites)   │   │
│  └──────┬───────┘   └──────────────┬───────────────┘   │
└─────────┼──────────────────────────┼────────────────────┘
          │ POST /index              │ POST /query (SSE)
          ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend                        │
│                                                         │
│  ┌─────────────┐   ┌──────────────┐  ┌──────────────┐  │
│  │  indexer.py │   │ retriever.py │  │   agent.py   │  │
│  │             │   │              │  │              │  │
│  │ gitpython   │   │ ChromaDB     │  │ Claude API   │  │
│  │ tiktoken    │   │ cosine sim   │  │ Streaming    │  │
│  │ MiniLM-L6   │   │ CrossEncoder │  │              │  │
│  └──────┬──────┘   └──────┬───────┘  └──────────────┘  │
│         │                 │                              │
│         ▼                 ▼                              │
│  ┌─────────────────────────────────┐                    │
│  │         ChromaDB (local)        │                    │
│  │    Persistent vector store      │                    │
│  └─────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- An Anthropic API key

### Backend

```bash
cd backend
cp ../.env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000`

---

## Example queries to try

After indexing `https://github.com/ShivaShanmukh/Job-Agent`:

- "How does the job agent handle retries?"
- "What external APIs does this project integrate with?"
- "Explain the main entry point and how it starts"
- "How are environment variables managed?"
- "What does the agent do when a job application fails?"

---

## Design decisions

### Chunking strategy
Files are chunked into **500-token segments with 50-token overlap** using tiktoken's `cl100k_base` encoding. The overlap ensures that semantically coherent passages that straddle chunk boundaries aren't lost. 500 tokens balances granularity (specific answers) vs. context (enough code to understand a chunk).

### Why ChromaDB
ChromaDB is an embedded vector store that runs in-process with zero infrastructure — no separate server needed for local dev. It uses HNSW for fast approximate nearest-neighbor search and persists to disk. For production at scale you'd swap it for Qdrant or Pinecone, but ChromaDB is ideal for this project's scope.

### Why streaming
Claude's streaming API delivers tokens as they're generated. This drastically improves perceived latency — users see the answer forming in real time rather than waiting 5–10 seconds for a complete response. The FastAPI backend uses SSE (Server-Sent Events) to pipe the stream directly to the browser.

### Citation approach
After streaming completes, the agent yields a `citations` event containing the unique file paths of all context chunks that were given to Claude. This is conservative but honest — we report which files Claude *had access to* rather than trying to parse which files it actually referenced (which is harder and error-prone).

### Reranking
Initial retrieval fetches 15 candidates via cosine similarity (fast but imprecise). A `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder then scores each (query, chunk) pair jointly — much more accurate than bi-encoder similarity alone — and the top 5 are sent to Claude. This two-stage approach gives both speed and quality.

---

## API reference

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| POST | `/index` | `{ github_url }` | Clone, chunk, and embed a repo |
| POST | `/query` | `{ query, repo_name }` | RAG query, returns SSE stream |
| GET | `/repos` | — | List all indexed repos |
| GET | `/health` | — | Health check |
