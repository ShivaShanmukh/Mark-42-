"""Pre-download the embedding + reranker models at build time.

Run during the Railway build (see backend/railway.json -> build.buildCommand).
Keeping this in a real file (instead of an inline `python -c "..."`) avoids the
shell/builder stripping the quotes around the model names — which previously
turned SentenceTransformer('all-MiniLM-L6-v2') into bare arithmetic and failed
the build with `NameError: name 'MiniLM' is not defined`.

Caching the models into the build image means the first request isn't slow and
the runtime container never needs network access to fetch them.
"""

from sentence_transformers import CrossEncoder, SentenceTransformer

if __name__ == "__main__":
    SentenceTransformer("all-MiniLM-L6-v2")
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("[build] embedding + reranker models cached")
