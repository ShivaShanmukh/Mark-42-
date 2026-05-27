import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent import answer
from indexer import index_repo, list_indexed_repos
from retriever import retrieve


def _validate_env() -> None:
    required = ["ANTHROPIC_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"[startup] FATAL: missing required env vars: {missing}", file=sys.stderr)
        sys.exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _validate_env()
    # Trigger model warm-up on startup so the first request isn't slow.
    # Models are already cached in the build image; this just loads them into RAM.
    from indexer import _embedder  # noqa: F401  warm up shared embedder
    from retriever import _reranker  # noqa: F401  warm up reranker
    yield


app = FastAPI(title="Codebase Intelligence", version="1.0.0", lifespan=lifespan)

_raw_origins = os.getenv("CORS_ORIGINS", "*")
_origins: list[str] = [o.strip() for o in _raw_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IndexRequest(BaseModel):
    github_url: str


class QueryRequest(BaseModel):
    query: str
    repo_name: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "Codebase Intelligence"}


@app.get("/repos")
def get_repos():
    try:
        repos = list_indexed_repos()
        return {"repos": repos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index")
def index(request: IndexRequest):
    if not request.github_url.startswith("https://github.com/"):
        raise HTTPException(status_code=400, detail="Only public GitHub URLs are supported")
    try:
        result = index_repo(request.github_url)
        if result["status"] == "error":
            raise HTTPException(status_code=422, detail=result["message"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        context = retrieve(request.query, request.repo_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Repo not found or retrieval failed: {e}")

    if not context:
        raise HTTPException(status_code=404, detail="No relevant chunks found for this query")

    async def event_generator():
        try:
            async for chunk in answer(request.query, context):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        },
    )
