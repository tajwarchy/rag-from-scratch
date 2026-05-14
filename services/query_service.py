"""
Query service — full retrieval-augmented generation pipeline.

On every query:
    1. Check embedding cache → embed query if miss
    2. Search FAISS index for top-k chunks
    3. Build a grounded prompt from retrieved chunks
    4. Call LLM service → return answer + sources + timing

This service only READS the FAISS index. It never modifies it.
All mutation lives in ingestion_service.py (Single Responsibility).
"""

import time

import core.cache as cache
from core.embedder import embed_query
from core.faiss_store import search, get_index_stats
from core.config_loader import load_config
from services.llm_service import generate


def query(question: str, top_k: int | None = None) -> dict:
    """
    Run the full RAG pipeline for a question.

    Returns:
        {
            "answer":            str,
            "sources":           list[dict],
            "cache_hit":         bool,
            "retrieval_time_ms": int,
            "llm_time_ms":       int,
        }

    Raises RuntimeError if no FAISS index is loaded.
    """
    cfg = load_config()
    if top_k is None:
        top_k = cfg["faiss"]["top_k"]

    stats = get_index_stats()
    if not stats["index_loaded"]:
        raise RuntimeError("No FAISS index loaded. Run POST /index first.")

    # ── Step 1: Embed query (cache-first) ────────────────────────────────────
    t0 = time.time()

    cached_vec = cache.get(question)
    cache_hit = cached_vec is not None

    if cache_hit:
        query_vec = cached_vec
    else:
        query_vec = embed_query(question)
        cache.set(question, query_vec)

    # ── Step 2: FAISS similarity search ─────────────────────────────────────
    chunks = search(query_vec, top_k=top_k)
    retrieval_time_ms = int((time.time() - t0) * 1000)

    # ── Step 3: Build grounded prompt ────────────────────────────────────────
    prompt = _build_prompt(question, chunks)

    # ── Step 4: Call LLM ─────────────────────────────────────────────────────
    t1 = time.time()
    answer = generate(prompt)
    llm_time_ms = int((time.time() - t1) * 1000)

    # Strip score from sources before returning (internal detail)
    sources = [
        {
            "chunk_id":    c["chunk_id"],
            "source_file": c["source_file"],
            "page":        c["page"],
            "text":        c["text"],
        }
        for c in chunks
    ]

    return {
        "answer":            answer,
        "sources":           sources,
        "cache_hit":         cache_hit,
        "retrieval_time_ms": retrieval_time_ms,
        "llm_time_ms":       llm_time_ms,
    }


# ── Prompt template ───────────────────────────────────────────────────────────

def _build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build a grounded QA prompt from retrieved chunks.

    Design principles:
    - Each chunk is labelled with its source so the model can attribute answers.
    - The model is explicitly told to answer ONLY from the provided context.
    - If the context doesn't contain the answer, the model says so rather than
      hallucinating — this is the core safety property of a grounded RAG system.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Source {i}: {chunk['source_file']}, page {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a precise question-answering assistant.
Answer the question using ONLY the context provided below.
If the answer is not present in the context, say "I cannot find the answer in the provided documents."
Do not use any external knowledge. Cite the source number (e.g. [Source 1]) when referencing information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    return prompt