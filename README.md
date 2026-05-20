# Codebase Intelligence 🔍

> Ask questions about any public GitHub repository in plain English and get accurate, grounded answers with source citations — powered by Claude Sonnet, ChromaDB, and a two-stage RAG pipeline.

---

## What is this?

Codebase Intelligence is a production-ready **Retrieval-Augmented Generation (RAG)** agent that turns any public GitHub repository into a searchable knowledge base you can chat with.

You paste a GitHub URL. It clones the repo, chunks every code file, embeds them with a local sentence transformer, and stores them in a vector database. You then ask natural language questions — *"How does authentication work?"*, *"Where are retries handled?"*, *"What does the scheduler do?"* — and get streaming answers that cite the exact files they came from.

No hallucinations about code that doesn't exist. No vague summaries. Just grounded answers from the actual source.

---

## Live Demo

**Frontend:** Deployed on Railway  
**Backend:** Deployed on Railway  
**Repo:** [github.com/ShivaShanmukh/Mark-42-](https://github.com/ShivaShanmukh/Mark-42-)

Default example pre-loaded: [ShivaShanmukh/Job-Agent](https://github.com/ShivaShanmukh/Job-Agent)

---

## Features

- **Index any public GitHub repo** — clones, walks, filters, chunks, and embeds in one click
- **Two-stage retrieval** — fast cosine similarity (top 15) followed by cross-encoder reranking (top 5) for high-precision results
- **Streaming answers** — tokens appear as Claude generates them, no waiting
- **Source citations** — every answer shows which files the context came from
- **Persistent vector store** — ChromaDB persists indexed repos to disk; no re-indexing needed between restarts
- **Multi-repo support** — index as many repos as you want; switch between them in the UI
- **Zero API key needed for embeddings** — uses `all-MiniLM-L6-v2` locally via sentence-transformers

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Browser (Next.js)                          │
│                                                                    │
│   ┌─────────────────────┐    ┌──────────────────────────────┐    │
│   │    Sidebar           │    │      Chat Interface           │    │
│   │  ┌───────────────┐  │    │                              │    │
│   │  │  GitHub URL   │  │    │  user: "How do retries work?"│    │
│   │  │  + Index Repo │  │    │                              │    │
│   │  └───────────────┘  │    │  assistant: "Based on        │    │
│   │  ┌───────────────┐  │    │  scheduler.py lines 42-87..."│    │
│   │  │  Chunk count  │  │    │                              │    │
│   │  │  Repo status  │  │    │  [Sources: scheduler.py,     │    │
│   │  └───────────────┘  │    │   main.py, config.py]        │    │
│   └─────────────────────┘    └──────────────────────────────┘    │
└────────────────────────┬─────────────────────────────────────────┘
                         │ /api/* (Next.js proxy rewrite)
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                               │
│                                                                    │
│  POST /index                POST /query              GET /repos   │
│       │                          │                                │
│       ▼                          ▼                                │
│  ┌──────────────┐      ┌──────────────────┐                      │
│  │  indexer.py  │      │  retriever.py    │                      │
│  │              │      │                  │                      │
│  │ 1. git clone │      │ 1. Embed query   │                      │
│  │ 2. Walk files│      │    (MiniLM-L6)   │                      │
│  │ 3. Chunk     │      │ 2. Cosine search │                      │
│  │    500 tokens│      │    top 15 chunks │                      │
│  │    50 overlap│      │ 3. CrossEncoder  │                      │
│  │ 4. Embed     │      │    rerank → top 5│                      │
│  │    (MiniLM)  │      └────────┬─────────┘                      │
│  │ 5. Store in  │               │                                │
│  │    ChromaDB  │               ▼                                │
│  └──────┬───────┘      ┌──────────────────┐                      │
│         │              │    agent.py       │                      │
│         ▼              │                  │                      │
│  ┌──────────────┐      │ Claude Sonnet    │                      │
│  │  ChromaDB    │      │ Streaming API    │                      │
│  │  (HNSW index)│      │                  │                      │
│  │  Persistent  │      │ → SSE token      │                      │
│  │  on disk     │      │   stream         │                      │
│  └──────────────┘      │ → citations      │                      │
│                        └──────────────────┘                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **LLM** | Claude Sonnet (`claude-sonnet-4-20250514`) | Best reasoning-to-cost ratio; native streaming |
| **Embeddings** | `all-MiniLM-L6-v2` (sentence-transformers) | Fast, local, no API key, 384-dim vectors |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Joint query-doc scoring; much more accurate than bi-encoder alone |
| **Vector DB** | ChromaDB | Embedded, zero-infra, HNSW index, persists to disk |
| **Backend** | FastAPI + uvicorn | Async-native, SSE streaming, automatic OpenAPI docs |
| **Frontend** | Next.js 15 + Tailwind CSS | React streaming, server-side proxy rewrites |
| **Tokenizer** | tiktoken (`cl100k_base`) | Same tokenizer as Claude for accurate chunk sizing |
| **Git cloning** | gitpython | Clean Python API for shallow clones |
| **Deployment** | Railway | Monorepo support, auto-deploy on push |

---

## Project Structure

```
rag-agent/
├── backend/
│   ├── main.py          # FastAPI app — routes, CORS, SSE streaming
│   ├── indexer.py       # Clone → chunk → embed → store in ChromaDB
│   ├── retriever.py     # Cosine search + CrossEncoder rerank
│   ├── agent.py         # Claude API RAG chain with streaming
│   ├── requirements.txt
│   └── railway.json     # Railway backend deployment config
├── frontend/
│   ├── src/app/
│   │   ├── page.tsx     # Main UI — sidebar + streaming chat
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── next.config.ts   # Proxy rewrite: /api/* → BACKEND_URL
│   ├── package.json
│   └── railway.json     # Railway frontend deployment config
├── .env.example
├── .gitignore
├── nixpacks.toml
├── Procfile
└── README.md
```

---

## API Reference

### `POST /index`
Clone and index a GitHub repository.

**Request:**
```json
{ "github_url": "https://github.com/owner/repo" }
```

**Response:**
```json
{
  "status": "success",
  "repo_name": "repo",
  "collection_name": "repo",
  "chunk_count": 142
}
```

---

### `POST /query`
Query an indexed repository. Returns a **Server-Sent Events** stream.

**Request:**
```json
{ "query": "How does authentication work?", "repo_name": "repo" }
```

**SSE stream:**
```
data: {"type": "token", "content": "Based on "}
data: {"type": "token", "content": "auth.py..."}
data: {"type": "citations", "files": ["auth.py", "middleware.py"]}
```

---

### `GET /repos`
List all indexed repositories.

**Response:**
```json
{
  "repos": [
    {
      "repo_name": "Job-Agent",
      "github_url": "https://github.com/ShivaShanmukh/Job-Agent",
      "chunk_count": 41
    }
  ]
}
```

---

### `GET /health`
Health check.

```json
{ "status": "ok", "service": "Codebase Intelligence" }
```

---

## Local Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- An [Anthropic API key](https://console.anthropic.com)

### 1. Clone the repo

```bash
git clone https://github.com/ShivaShanmukh/Mark-42-.git
cd Mark-42-
```

### 2. Backend

```bash
cd backend
cp ../.env.example .env
# Open .env and add your ANTHROPIC_API_KEY

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000`

---

## Environment Variables

### Backend
| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Your Anthropic API key from console.anthropic.com |

### Frontend
| Variable | Required | Description |
|---|---|---|
| `BACKEND_URL` | ✅ (production) | Full URL of the deployed backend, e.g. `https://xxx.up.railway.app`. Defaults to `http://localhost:8000` in dev. |

> **Note:** `BACKEND_URL` is a server-side env var read at runtime by Next.js rewrites — not a `NEXT_PUBLIC_` build-time var. You can change it in Railway without rebuilding the frontend.

---

## Deploying to Railway

This repo is set up for Railway monorepo deployment with two services.

### Backend service
- **Root Directory:** `backend`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Env vars:** `ANTHROPIC_API_KEY`
- **Health check:** `GET /health`

### Frontend service
- **Root Directory:** `frontend`
- **Start Command:** `npm run start`
- **Env vars:** `BACKEND_URL=https://<your-backend>.up.railway.app`

Railway auto-deploys both services on every push to `main`.

---

## Design Decisions

### Chunking strategy — 500 tokens, 50-token overlap
Files are chunked using `tiktoken` with the `cl100k_base` encoding (the same tokenizer Claude uses). 500 tokens gives enough context for Claude to understand a code segment without overwhelming the prompt. The 50-token overlap ensures that semantically meaningful passages that straddle chunk boundaries aren't split in half and lost.

### Two-stage retrieval — cosine + CrossEncoder reranking
Basic vector similarity (bi-encoder) is fast but imprecise — it compares query and document embeddings independently. The CrossEncoder scores each (query, chunk) pair jointly, which is far more accurate. We retrieve 15 candidates via cosine similarity first (fast), then rerank with the CrossEncoder and return the top 5 (accurate). This two-stage approach gives the speed of approximate search with near-reranker quality.

### Why ChromaDB
ChromaDB runs embedded in the same process as FastAPI — no separate database server to manage or pay for. It uses HNSW under the hood for fast approximate nearest-neighbour search, and persists the index to disk automatically. For a project at this scale it's the right tool; at millions of chunks you'd swap it for Qdrant or Pinecone.

### Why streaming
Claude's API delivers tokens as they're generated. Using SSE (Server-Sent Events), the FastAPI backend pipes each token directly to the browser as it arrives. Users see the answer forming in real time instead of staring at a spinner for 8–10 seconds. FastAPI's `StreamingResponse` handles this natively with an async generator.

### Citation approach
After streaming completes, the backend yields a final `citations` event listing the file paths of every chunk that was passed to Claude as context. This is intentionally conservative — we report which files Claude *had access to*, not which ones it claims to have used. Parsing Claude's own citations from the prose is fragile and error-prone; this approach is reliable and honest.

### Proxy rewrite instead of NEXT_PUBLIC_ env var
`NEXT_PUBLIC_*` variables are embedded into the JS bundle at build time. On Railway, the backend URL isn't known until after the backend deploys, which creates a chicken-and-egg problem. The solution: the frontend always calls relative `/api/*` paths, and `next.config.ts` rewrites them to `${BACKEND_URL}/*` server-side. `BACKEND_URL` is a plain server env var read at process startup — changeable without a rebuild.

---

## Example queries to try

After indexing `https://github.com/ShivaShanmukh/Job-Agent`:

- *"How does the job agent handle retries?"*
- *"What external APIs does this project integrate with?"*
- *"Explain the main entry point and what arguments it accepts"*
- *"How are environment variables and config managed?"*
- *"What does the scheduler do and how often does it run?"*
- *"How does email notification work?"*
- *"Walk me through what happens when I run python main.py"*

---

## Security

- `.env` is in `.gitignore` — your API key is never committed
- Only public GitHub repos are accepted (`https://github.com/` prefix enforced)
- CORS is open in dev; lock `allow_origins` to your frontend domain in production
- Next.js upgraded to 15.5.18 (0 CVEs as of May 2026)
- PostCSS pinned to `^8.5.10` via `overrides` to patch GHSA-qx2v-qp2m-jg93

---

## Built with

- [Anthropic Claude API](https://docs.anthropic.com)
- [ChromaDB](https://docs.trychroma.com)
- [sentence-transformers](https://www.sbert.net)
- [FastAPI](https://fastapi.tiangolo.com)
- [Next.js](https://nextjs.org)
- [Railway](https://railway.app)

---

*Built by [Shiva Shanmukh](https://github.com/ShivaShanmukh)*
