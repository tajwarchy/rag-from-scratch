"""
FastAPI route handlers — thin wrappers over service modules.
Each handler does three things only: validate input, call a service, return a response.
No business logic lives here — that belongs in the service layer.
"""

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.models import (
    IndexRequest, IndexResponse,
    QueryRequest, QueryResponse,
    HealthResponse,
)
from services.ingestion_service import run_ingestion, get_status
from services.query_service import query as run_query
from core.faiss_store import get_index_stats
import core.cache as cache
from core.config_loader import load_config

router = APIRouter()


# ── POST /index ───────────────────────────────────────────────────────────────

@router.post("/index", response_model=IndexResponse, status_code=202)
async def index(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Start ingestion as a background task and return immediately (202 Accepted).

    Why async / background?
    Ingestion reads many files, chunks, and batch-embeds them — potentially
    minutes of work. Blocking the HTTP thread for that duration is a
    production anti-pattern. BackgroundTasks is the minimal correct model;
    in production this would be a Celery or ARQ job queue.
    """
    if not Path(request.pdf_dir).exists():
        raise HTTPException(
            status_code=422,
            detail=f"PDF directory '{request.pdf_dir}' does not exist.",
        )

    valid_strategies = {"fixed", "sentence", "sliding"}
    if request.chunking_strategy not in valid_strategies:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid strategy '{request.chunking_strategy}'. Choose: {sorted(valid_strategies)}",
        )

    background_tasks.add_task(
        run_ingestion,
        request.pdf_dir,
        request.chunking_strategy,
    )

    return IndexResponse(
        status="indexing_started",
        pdf_dir=request.pdf_dir,
        chunking_strategy=request.chunking_strategy,
        message="Indexing running in background. Poll GET /health for completion.",
    )


# ── POST /query ───────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Synchronous RAG query — embeds question, retrieves chunks, calls LLM.
    Returns only after the full pipeline completes so the client gets the answer
    in a single round-trip.
    """
    try:
        result = run_query(request.question, top_k=request.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return QueryResponse(**result)


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health():
    """
    Observability endpoint — index stats + cache metrics + uptime.
    Use this to confirm indexing completed and to monitor cache performance.
    """
    from main import get_uptime  # imported here to avoid circular import at module load

    index_stats    = get_index_stats()
    cache_stats    = cache.stats()
    ingestion_info = get_status()

    return HealthResponse(
        status="ok",
        index_loaded=index_stats["index_loaded"],
        index_size=index_stats["index_size"],
        chunk_count=index_stats["chunk_count"],
        chunking_strategy=ingestion_info.get("chunking_strategy"),
        cache_hits=cache_stats["cache_hits"],
        cache_misses=cache_stats["cache_misses"],
        cache_hit_rate=cache_stats["cache_hit_rate"],
        uptime_seconds=get_uptime(),
    )