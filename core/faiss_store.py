"""
FAISS store — build, persist, load, and search the vector index.

Why save the index to disk?
  Embedding 300–500 chunks takes ~10–30 seconds even on MPS.
  Without persistence, every server restart would re-embed everything.
  Saving to disk means the server loads in <1 second on subsequent starts.
  Tradeoff: the system is now stateful — the index on disk must stay in
  sync with the metadata file. If you delete one, delete both.

Index type: IndexFlatL2
  Exact nearest-neighbour search. No approximation, no training required.
  Correct choice for <100k vectors. For larger corpora, swap to
  IndexIVFFlat (requires a training step) — the interface stays the same.
"""

import json
import numpy as np
import faiss
from pathlib import Path

from core.config_loader import load_config

# Module-level state — populated by build_index() or load_index()
_index: faiss.Index | None = None
_chunks: list[dict] = []          # chunk metadata parallel to index vectors


def build_index(embeddings: np.ndarray, chunks: list[dict]) -> None:
    """
    Build a FAISS index from embeddings and save it to disk.
    Also caches index + chunks in module-level state for immediate querying.

    Args:
        embeddings: float32 array of shape (n, dim)
        chunks:     list of chunk dicts (same order as embeddings)
    """
    global _index, _chunks
    cfg = load_config()

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Persist index binary
    index_path = Path(cfg["paths"]["index_file"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"[FAISS] Index saved → {index_path}  ({index.ntotal} vectors, dim={dim})")

    # Persist metadata (already written by ingestion_service, but refresh here
    # to guarantee alignment with the embeddings just built)
    metadata_path = Path(cfg["paths"]["metadata_file"])
    with open(metadata_path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"[FAISS] Metadata saved → {metadata_path}  ({len(chunks)} chunks)")

    _index = index
    _chunks = chunks


def load_index() -> bool:
    """
    Load FAISS index + chunk metadata from disk into module-level state.
    Returns True if loaded successfully, False if no index exists yet.
    Called at server startup (Phase 5).
    """
    global _index, _chunks
    cfg = load_config()

    index_path    = Path(cfg["paths"]["index_file"])
    metadata_path = Path(cfg["paths"]["metadata_file"])

    if not index_path.exists() or not metadata_path.exists():
        print("[FAISS] No existing index found on disk — run POST /index first.")
        return False

    _index = faiss.read_index(str(index_path))
    with open(metadata_path, "r") as f:
        _chunks = json.load(f)

    print(f"[FAISS] Index loaded from disk: {_index.ntotal} vectors, {len(_chunks)} chunks")
    return True


def search(query_vec: np.ndarray, top_k: int) -> list[dict]:
    """
    Find the top_k most similar chunks to the query vector.

    Returns a list of chunk dicts enriched with a 'score' field
    (L2 distance — lower is more similar).

    Raises RuntimeError if no index is loaded.
    """
    if _index is None:
        raise RuntimeError("No FAISS index loaded. Run POST /index first.")

    distances, indices = _index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:           # FAISS returns -1 when fewer than top_k results exist
            continue
        chunk = dict(_chunks[idx])
        chunk["score"] = float(dist)
        results.append(chunk)

    return results


def get_index_stats() -> dict:
    """Return index stats for the /health endpoint."""
    return {
        "index_loaded": _index is not None,
        "index_size":   _index.ntotal if _index is not None else 0,
        "chunk_count":  len(_chunks),
    }