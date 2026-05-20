import json
import os
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

app = FastAPI(title="Codebase Intelligence", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
