# API Specification — Document Q&A RAG System

> Contract defined before implementation. All FastAPI routes must match these schemas exactly.

---

## Base URL
```
http://localhost:8000
```

---

## Endpoints

### 1. `POST /index`

Ingests a folder of PDFs — extracts, chunks, embeds, and saves the FAISS index to disk.
This is an **asynchronous** operation: the endpoint returns immediately and processing
runs in the background. This models a real production pattern where indexing is a
long-running job and should not block the HTTP thread.

**Request body**
```json
{
  "pdf_dir": "data/pdfs",
  "chunking_strategy": "fixed"
}
```

| Field               | Type   | Required | Values                               |
|---------------------|--------|----------|--------------------------------------|
| `pdf_dir`           | string | yes      | Path to directory containing PDFs    |
| `chunking_strategy` | string | no       | `"fixed"` (default), `"sentence"`, `"sliding"` |

**Response — 202 Accepted**
```json
{
  "status": "indexing_started",
  "pdf_dir": "data/pdfs",
  "chunking_strategy": "fixed",
  "message": "Indexing running in background. Poll /health for completion."
}
```

**Error — 422 Unprocessable Entity** (directory not found)
```json
{
  "detail": "PDF directory 'data/pdfs' does not exist."
}
```

---

### 2. `POST /query`

Embeds the question, checks the embedding cache, retrieves the top-k chunks from FAISS,
builds a grounded prompt, calls the LLM, and returns the answer.
This is a **synchronous** operation: the response is returned only after the full
retrieval + generation pipeline completes.

**Request body**
```json
{
  "question": "What is the attention mechanism in transformers?",
  "top_k": 5
}
```

| Field      | Type    | Required | Default | Constraints      |
|------------|---------|----------|---------|------------------|
| `question` | string  | yes      | —       | 1–1000 chars     |
| `top_k`    | integer | no       | 5       | 1–20             |

**Response — 200 OK**
```json
{
  "answer": "The attention mechanism allows the model to...",
  "sources": [
    {
      "chunk_id": 42,
      "source_file": "attention_paper.pdf",
      "page": 3,
      "text": "We propose a new simple network architecture..."
    }
  ],
  "cache_hit": false,
  "retrieval_time_ms": 12,
  "llm_time_ms": 3420
}
```

| Field               | Type    | Description                                      |
|---------------------|---------|--------------------------------------------------|
| `answer`            | string  | LLM-generated answer grounded in retrieved chunks|
| `sources`           | array   | Top-k chunks used to build the prompt            |
| `cache_hit`         | boolean | Whether the query embedding was served from cache|
| `retrieval_time_ms` | integer | Time spent on embedding + FAISS search           |
| `llm_time_ms`       | integer | Time spent on LLM generation                     |

**Error — 400 Bad Request** (index not loaded)
```json
{
  "detail": "No FAISS index loaded. Run POST /index first."
}
```

---

### 3. `GET /health`

Observability endpoint. Returns the current system state including index stats
and cache performance. Use this to monitor the system and confirm indexing completed.

**Response — 200 OK**
```json
{
  "status": "ok",
  "index_loaded": true,
  "index_size": 1024,
  "chunk_count": 1024,
  "chunking_strategy": "fixed",
  "cache_hits": 18,
  "cache_misses": 32,
  "cache_hit_rate": 0.36,
  "uptime_seconds": 142
}
```

| Field               | Type    | Description                                       |
|---------------------|---------|---------------------------------------------------|
| `status`            | string  | Always `"ok"` if the service is running           |
| `index_loaded`      | boolean | Whether a FAISS index is currently in memory      |
| `index_size`        | integer | Number of vectors in the FAISS index              |
| `chunk_count`       | integer | Number of text chunks stored in metadata          |
| `chunking_strategy` | string  | Strategy used when the current index was built    |
| `cache_hits`        | integer | Cumulative embedding cache hits since startup     |
| `cache_misses`      | integer | Cumulative embedding cache misses since startup   |
| `cache_hit_rate`    | float   | `hits / (hits + misses)`, 0.0 if no queries yet  |
| `uptime_seconds`    | integer | Seconds since the FastAPI server started          |

---

## Design Notes

### Sync vs async — why?
- **`/index` is async** because ingestion involves reading many files, chunking, and
  batch-embedding — potentially minutes of work. Blocking the HTTP thread for that
  duration is a production anti-pattern. A background task is the minimal correct model.
- **`/query` is sync** because users expect an answer immediately. Latency is measured
  and returned in the response so bottlenecks are visible.

### Single responsibility
The ingestion pipeline (extract → chunk → embed → index) and the query pipeline
(embed → cache → retrieve → generate) are implemented as separate service modules.
They share the FAISS index on disk but have no runtime coupling.

### LLM abstraction
The LLM is called through a single `generate()` function in `services/llm_service.py`.
Swapping from Ollama to OpenAI requires changing only that one function — the query
service has no knowledge of which LLM backend is in use.