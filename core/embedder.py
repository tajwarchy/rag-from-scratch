"""
Embedder — wraps Sentence-Transformers for chunk and query embedding.

Device strategy (M1 MacBook Air):
  - Uses MPS (Apple Silicon GPU) when available for batch embedding.
  - Falls back to CPU if MPS is unavailable.
  - num_workers=0 throughout — macOS multiprocessing constraint with PyTorch.

The model (all-MiniLM-L6-v2) produces 384-dimensional embeddings.
It is loaded once as a module-level singleton to avoid reloading on every call.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from core.config_loader import load_config

_model = None  # singleton — loaded once on first call


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        cfg = load_config()
        model_name = cfg["embedding"]["model_name"]
        device = _resolve_device(cfg["embedding"]["device"])
        print(f"[Embedder] Loading model '{model_name}' on device '{device}'...")
        _model = SentenceTransformer(model_name, device=device)
        print(f"[Embedder] Model loaded. Embedding dim: {_model.get_sentence_embedding_dimension()}")
    return _model


def _resolve_device(preferred: str) -> str:
    """Use MPS if available and preferred, else CPU."""
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """
    Embed a list of chunk dicts.
    Returns a float32 numpy array of shape (n_chunks, embedding_dim).

    Processes in batches (batch_size from config).
    """
    cfg = load_config()
    batch_size = cfg["embedding"]["batch_size"]
    model = _get_model()

    texts = [c["text"] for c in chunks]
    total = len(texts)
    print(f"[Embedder] Embedding {total} chunks in batches of {batch_size}...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalise — dot product == cosine similarity
    )

    embeddings = embeddings.astype(np.float32)
    print(f"[Embedder] Done. Shape: {embeddings.shape}")
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.
    Returns a float32 numpy array of shape (1, embedding_dim).

    Called at query time — result is cached by the query service
    to avoid redundant computation on repeated queries.
    """
    model = _get_model()
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec.astype(np.float32)