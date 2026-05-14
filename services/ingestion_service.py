"""
Ingestion service — orchestrates the full pipeline:
    PDF directory → extract → chunk → embed → FAISS index → disk

System design note:
    This service is intentionally separate from query_service.py.
    Single Responsibility: ingestion mutates the index; querying reads it.
    They share the FAISS index on disk but have zero runtime coupling.

    This step is designed to be called asynchronously (via FastAPI
    BackgroundTasks in Phase 5) because it can take minutes for large
    PDF collections. It should never block an HTTP thread.
"""

import time

from core.extractor import extract_text_from_dir
from core.chunker import get_chunker
from core.embedder import embed_chunks
from core.faiss_store import build_index
from core.config_loader import load_config

# Shared state — written here, read by query_service and /health
_ingestion_status = {
    "state":             "idle",   # idle | running | done | error
    "chunk_count":       0,
    "chunking_strategy": None,
    "last_run_seconds":  None,
    "error":             None,
}


def get_status() -> dict:
    """Return a copy of the current ingestion status (for /health)."""
    return dict(_ingestion_status)


def run_ingestion(pdf_dir: str, strategy: str) -> list[dict]:
    """
    Full ingestion pipeline: extract → chunk → embed → index → persist.

    Returns the list of chunk dicts.
    Updates _ingestion_status throughout so /health can report progress.
    """
    global _ingestion_status
    cfg = load_config()

    _ingestion_status.update({
        "state":             "running",
        "chunking_strategy": strategy,
        "error":             None,
    })

    try:
        t_start = time.time()

        # ── Step 1: Extract ──────────────────────────────────────────────────
        print(f"\n[Ingestion] Extracting PDFs from: {pdf_dir}")
        pages = extract_text_from_dir(pdf_dir)
        print(f"[Ingestion] Extracted {len(pages)} pages total")

        # ── Step 2: Chunk ────────────────────────────────────────────────────
        print(f"[Ingestion] Chunking with strategy: '{strategy}'")
        chunker = get_chunker(strategy)
        chunks = chunker(pages, cfg)
        print(f"[Ingestion] Created {len(chunks)} chunks")

        # ── Step 3: Embed ────────────────────────────────────────────────────
        print(f"[Ingestion] Embedding chunks...")
        embeddings = embed_chunks(chunks)

        # ── Step 4: Build & persist FAISS index ─────────────────────────────
        print(f"[Ingestion] Building FAISS index...")
        build_index(embeddings, chunks)

        elapsed = round(time.time() - t_start, 2)
        _ingestion_status.update({
            "state":            "done",
            "chunk_count":      len(chunks),
            "last_run_seconds": elapsed,
        })
        print(f"[Ingestion] Pipeline complete in {elapsed}s")

        return chunks

    except Exception as e:
        _ingestion_status.update({"state": "error", "error": str(e)})
        print(f"[Ingestion] ERROR: {e}")
        raise