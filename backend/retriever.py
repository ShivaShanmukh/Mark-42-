from sentence_transformers import CrossEncoder

from indexer import _embedder, get_collection

_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

TOP_K_RETRIEVE = 15
TOP_K_RETURN = 5


def retrieve(query: str, repo_name: str) -> list[dict]:
    collection = get_collection(repo_name)

    count = collection.count()
    if count == 0:
        return []

    query_embedding = _embedder.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K_RETRIEVE, count),
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    pairs = [[query, doc] for doc in docs]
    scores = _reranker.predict(pairs).tolist()

    ranked = sorted(
        zip(docs, metas, distances, scores),
        key=lambda x: x[3],
        reverse=True,
    )

    return [
        {
            "text": doc,
            "file_path": meta.get("file_path", ""),
            "line_start": meta.get("line_start", 0),
            "line_end": meta.get("line_end", 0),
            "chunk_index": meta.get("chunk_index", 0),
            "similarity": 1 - dist,
            "rerank_score": score,
        }
        for doc, meta, dist, score in ranked[:TOP_K_RETURN]
    ]
