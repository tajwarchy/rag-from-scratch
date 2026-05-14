"""
Pydantic request/response models — must match api_spec.md exactly.
Defined here and imported by routes.py to keep route handlers thin.
"""

from pydantic import BaseModel, Field


# ── /index ────────────────────────────────────────────────────────────────────

class IndexRequest(BaseModel):
    pdf_dir: str = Field(
        default="data/pdfs",
        description="Path to directory containing PDF files.",
    )
    chunking_strategy: str = Field(
        default="fixed",
        description="Chunking strategy to use: fixed | sentence | sliding",
    )


class IndexResponse(BaseModel):
    status: str
    pdf_dir: str
    chunking_strategy: str
    message: str


# ── /query ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to answer from indexed documents.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve from FAISS.",
    )


class ChunkSource(BaseModel):
    chunk_id: int
    source_file: str
    page: int
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[ChunkSource]
    cache_hit: bool
    retrieval_time_ms: int
    llm_time_ms: int


# ── /health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    index_size: int
    chunk_count: int
    chunking_strategy: str | None
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    uptime_seconds: int