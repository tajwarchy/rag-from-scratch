"""
In-memory query embedding cache — simulates Redis behaviour using a plain dict.

Why cache query embeddings?
  Embedding a query takes ~5–15ms on MPS. For repeated queries (e.g. a user
  asking the same question, or the load test in Phase 6), re-embedding wastes
  compute. This cache makes repeated queries near-instant.

Key   = SHA-256 hash of the query string (normalised to lowercase + stripped).
Value = numpy float32 array of shape (1, embedding_dim).

To upgrade to Redis in production:
  Replace the dict + OrderedDict below with:
      import redis, pickle
      _r = redis.Redis(host="localhost", port=6379)
      def get(key): v = _r.get(key); return pickle.loads(v) if v else None
      def set(key, val): _r.setex(key, 3600, pickle.dumps(val))
  The cache interface (get/set/stats) stays identical — no changes elsewhere.
"""

import hashlib
from collections import OrderedDict

import numpy as np

from core.config_loader import load_config

# OrderedDict preserves insertion order → enables LRU-style eviction
_cache: OrderedDict[str, np.ndarray] = OrderedDict()
_hits   = 0
_misses = 0


def _make_key(query: str) -> str:
    normalised = query.strip().lower()
    return hashlib.sha256(normalised.encode()).hexdigest()


def get(query: str) -> np.ndarray | None:
    """Return cached embedding for query, or None on cache miss."""
    global _hits, _misses
    key = _make_key(query)
    if key in _cache:
        _cache.move_to_end(key)   # mark as recently used
        _hits += 1
        return _cache[key]
    _misses += 1
    return None


def set(query: str, embedding: np.ndarray) -> None:
    """Store embedding in cache. Evicts oldest entry if max_size exceeded."""
    cfg = load_config()
    max_size = cfg["cache"]["max_size"]
    key = _make_key(query)
    _cache[key] = embedding
    _cache.move_to_end(key)
    if len(_cache) > max_size:
        _cache.popitem(last=False)   # evict oldest


def stats() -> dict:
    """Return cache metrics for the /health endpoint."""
    total = _hits + _misses
    return {
        "cache_hits":     _hits,
        "cache_misses":   _misses,
        "cache_hit_rate": round(_hits / total, 3) if total > 0 else 0.0,
    }