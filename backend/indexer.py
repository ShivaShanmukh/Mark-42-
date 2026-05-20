import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import chromadb
import tiktoken
from git import Repo
from sentence_transformers import SentenceTransformer

SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".json"}
CHUNK_SIZE = 500  # tokens
OVERLAP = 50      # tokens
EMBED_MODEL = "all-MiniLM-L6-v2"

_encoder = tiktoken.get_encoding("cl100k_base")
_embedder = SentenceTransformer(EMBED_MODEL)
_chroma_client = chromadb.PersistentClient(path="./chroma_db")


def _tokenize(text: str) -> list[int]:
    return _encoder.encode(text)


def _chunk_text(text: str, file_path: str) -> list[dict]:
    tokens = _tokenize(text)
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _encoder.decode(chunk_tokens)

        # Estimate line range based on character position
        char_start = len(_encoder.decode(tokens[:start]))
        char_end = char_start + len(chunk_text)
        lines_before = text[:char_start].count("\n")
        lines_in_chunk = chunk_text.count("\n")

        chunks.append({
            "text": chunk_text,
            "file_path": file_path,
            "line_start": lines_before + 1,
            "line_end": lines_before + lines_in_chunk + 1,
            "chunk_index": chunk_index,
        })

        start += CHUNK_SIZE - OVERLAP
        chunk_index += 1

    return chunks


def _walk_repo(repo_dir: str) -> Generator[tuple[str, str], None, None]:
    repo_path = Path(repo_dir)
    for file_path in repo_path.rglob("*"):
        if file_path.is_file() and file_path.suffix in SUPPORTED_EXTENSIONS:
            # Skip hidden dirs and common noise
            parts = file_path.parts
            if any(p.startswith(".") for p in parts) or "node_modules" in parts:
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if content.strip():
                    rel_path = str(file_path.relative_to(repo_path))
                    yield rel_path, content
            except Exception:
                continue


def index_repo(github_url: str) -> dict:
    repo_name = github_url.rstrip("/").split("/")[-1]
    collection_name = repo_name.replace("-", "_").replace(".", "_")[:63]

    # Drop existing collection for fresh index
    try:
        _chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = _chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "github_url": github_url, "repo_name": repo_name},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        Repo.clone_from(github_url, tmpdir, depth=1)

        all_chunks = []
        for rel_path, content in _walk_repo(tmpdir):
            chunks = _chunk_text(content, rel_path)
            all_chunks.extend(chunks)

        if not all_chunks:
            return {"status": "error", "message": "No supported files found", "chunk_count": 0}

        # Batch embed
        texts = [c["text"] for c in all_chunks]
        embeddings = _embedder.encode(texts, batch_size=64, show_progress_bar=False).tolist()

        batch_size = 500
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            collection.add(
                ids=[f"{c['file_path']}::{c['chunk_index']}" for c in batch],
                embeddings=batch_embeddings,
                documents=[c["text"] for c in batch],
                metadatas=[
                    {
                        "file_path": c["file_path"],
                        "line_start": c["line_start"],
                        "line_end": c["line_end"],
                        "chunk_index": c["chunk_index"],
                        "repo_name": repo_name,
                    }
                    for c in batch
                ],
            )

    return {
        "status": "success",
        "repo_name": repo_name,
        "collection_name": collection_name,
        "chunk_count": len(all_chunks),
    }


def list_indexed_repos() -> list[dict]:
    collections = _chroma_client.list_collections()
    repos = []
    for col in collections:
        c = _chroma_client.get_collection(col.name)
        meta = c.metadata or {}
        repos.append({
            "collection_name": col.name,
            "repo_name": meta.get("repo_name", col.name),
            "github_url": meta.get("github_url", ""),
            "chunk_count": c.count(),
        })
    return repos


def get_collection(repo_name: str):
    collection_name = repo_name.replace("-", "_").replace(".", "_")[:63]
    return _chroma_client.get_collection(collection_name)
