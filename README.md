# Document Q&A — RAG from Scratch (FAISS)

A production-thinking retrieval-augmented generation pipeline built **without LangChain** — every component is implemented from scratch so the internals are fully visible. Embeds documents with Sentence-Transformers, indexes them in FAISS, and answers questions with Mistral 7B running locally via Ollama.

**Stack:** FastAPI · FAISS · Sentence-Transformers · PyMuPDF · Ollama (Mistral 7B) · Python 3.11  
**Environment:** MacBook Air M1 — MPS embedding, no GPU required for inference  
**No API keys. No cloud. Runs entirely on your machine.**

---

## Architecture

```
Client
  │
  ▼
FastAPI Gateway  (main.py)
  │
  ├── POST /index  ──▶  Ingestion Service  ──▶  Extractor  ──▶  Chunker
  │                           │                                      │
  │                           ▼                                      ▼
  │                       Embedder  ─────────────────────▶  FAISS Index ──▶ Disk
  │
  └── POST /query  ──▶  Query Service
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
               Cache hit?           Embedder
               (dict/Redis)             │
                    │                   ▼
                    └────────▶  FAISS Search (top-k)
                                        │
                                        ▼
                                  Prompt Builder
                                        │
                                        ▼
                                  LLM Service  ──▶  Ollama (Mistral 7B)
                                        │
                                        ▼
                                     Response
                              (answer + sources + timing)
```

**Data flow — ingestion:**
```
PDF files  ──▶  PyMuPDF extractor  ──▶  Chunker  ──▶  Sentence-Transformers  ──▶  FAISS index  ──▶  Disk
```

---

## Project Structure

```
rag-scratch/
├── main.py                    # FastAPI entrypoint + startup
├── config.yaml                # All parameters — no hardcoded values
├── api_spec.md                # API contract defined before any code
├── api/
│   ├── models.py              # Pydantic request/response schemas
│   └── routes.py              # Route handlers — thin wrappers over services
├── core/
│   ├── extractor.py           # PyMuPDF PDF text extraction + cleaning
│   ├── chunker.py             # Three chunking strategies
│   ├── embedder.py            # Sentence-Transformers wrapper (MPS)
│   ├── faiss_store.py         # FAISS index build, persist, load, search
│   ├── cache.py               # In-memory query embedding cache
│   └── config_loader.py       # YAML config singleton
├── services/
│   ├── ingestion_service.py   # Orchestrates extract → chunk → embed → index
│   ├── query_service.py       # Orchestrates cache → embed → retrieve → generate
│   └── llm_service.py         # LLM abstraction layer (Ollama / swap to OpenAI)
├── tests/
│   ├── ground_truth.json      # 15 QA pairs for retrieval evaluation
│   ├── eval_retrieval.py      # Recall@5 + MRR benchmark
│   └── eval_latency.py        # 50-query latency simulation
├── scripts/
│   └── download_pdfs.py       # Downloads 3 ArXiv papers as test corpus
├── data/pdfs/                 # PDF source documents (gitignored)
└── indexes/                   # FAISS index + metadata (gitignored)
```

---

## Quickstart

```bash
# 1. Clone and create environment
git clone https://github.com/tajwarchy/rag-scratch.git
cd rag-scratch
conda env create -f environment.yml
conda activate rag-scratch

# 2. Install and start Ollama
brew install ollama
ollama pull mistral
ollama serve &

# 3. Download test corpus (3 ArXiv papers)
python scripts/download_pdfs.py

# 4. Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 5. Index the documents
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"pdf_dir": "data/pdfs", "chunking_strategy": "sentence"}'

# 6. Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is multi-head attention?", "top_k": 5}'
```

Interactive API docs available at `http://localhost:8000/docs`.

---

## API Endpoints

| Method | Endpoint  | Description | Sync/Async |
|--------|-----------|-------------|------------|
| `POST` | `/index`  | Ingest a PDF directory | Async (202) |
| `POST` | `/query`  | Ask a question | Sync (200) |
| `GET`  | `/health` | Index stats + cache metrics + uptime | Sync (200) |

Full request/response schemas are documented in [`api_spec.md`](api_spec.md).

---

## Chunking Strategies

Three strategies are implemented in `core/chunker.py`, each with the same interface so they are fully interchangeable:

| Strategy | Description |
|----------|-------------|
| `fixed` | Fixed token-size windows with overlap (tiktoken). Predictable chunk sizes. |
| `sentence` | Groups sentences up to a max count. Respects natural language boundaries. |
| `sliding` | Sliding token window with configurable step size. Maximum context overlap. |

---

## Evaluation Results

### Retrieval — Recall@5 + MRR (15 ground-truth questions, 3 ArXiv papers)

| Strategy | Chunks | Recall@5 | MRR | Hits | Index time |
|----------|--------|----------|-----|------|------------|
| fixed | 198 | 0.867 | 0.658 | 13/15 | 7.0s |
| **sentence** | 371 | **0.933** | 0.628 | **14/15** | 3.24s |
| sliding | 362 | 0.867 | 0.608 | 13/15 | 3.48s |

**Winner: sentence-aware chunking** — 93.3% recall@5, hitting 14 of 15 questions.

Sentence-aware chunking outperforms fixed and sliding here because the test corpus is academic text with clear sentence structure. Sentence boundaries align with semantic units — an answer about "multi-head attention" rarely spans across a natural sentence break. Fixed and sliding strategies can split a relevant fact across two chunks, causing a retrieval miss even when the information is present.

Sliding window produces the most chunks (more overlap) but does not outperform sentence-aware, suggesting the bottleneck is semantic boundary alignment rather than coverage density.

### Latency — 50-query load simulation

| Metric | Retrieval | LLM | Total |
|--------|-----------|-----|-------|
| Mean | 391ms* | 20,086ms | 20,477ms |
| Median | 12ms | 19,754ms | 20,022ms |
| P95 | ~1,000ms | ~34,000ms | ~34,500ms |
| Min | 0ms | 9,641ms | 10,506ms |
| Max | 4,288ms | 39,675ms | 39,727ms |

*Retrieval mean is skewed by the first 20 cold-start queries (700–4,288ms). Cache hits reduce retrieval to 0–16ms.

**Cache hit rate: 60% (30/50 queries)** — the second half of the query set consisted of repeated questions, confirming the embedding cache eliminates redundant embedding computation entirely.

**Bottleneck: LLM generation accounts for ~98% of total latency.**

FAISS search on 198–371 vectors is effectively free (<5ms). The embedding step on MPS costs 300–4,000ms on the first call but collapses to <16ms on cache hits. Mistral 7B via Ollama on CPU is the dominant cost at 10–40 seconds per query.

**What would break first at 100 concurrent users:** The LLM. Ollama serves one request at a time by default — concurrent queries would queue. The fix is a request queue (e.g. Redis + workers) with multiple Ollama instances, or replacing Ollama with a served model endpoint (vLLM, TGI) that supports batched inference. FAISS at this scale is not a concern; it handles concurrent reads safely.

---

## System Design Notes

### Why is the FAISS index saved to disk?
Embedding 200–400 chunks takes 3–7 seconds on M1 MPS. Without persistence, every server restart re-embeds the entire corpus from scratch. Saving the binary index to disk means the server loads in under a second on subsequent starts. The tradeoff is statefulness: the index file and the metadata JSON must stay in sync. If you re-index with a different chunking strategy, both files are overwritten together.

### Why is ingestion separated from querying?
Single Responsibility. Ingestion mutates the index (write path). Querying reads it (read path). Separating them means you can re-index a new document corpus without touching the query pipeline, and scale the two independently. In production this separation maps directly to separate services — an ingestion worker and a query API — each with its own scaling policy.

### What would break first at 100 concurrent users?
The LLM inference step. FAISS handles concurrent reads without locking. The embedding model on MPS is fast enough that the cache absorbs most repeat queries. But Ollama processes one generation request at a time. At 100 users, generation requests would queue and latency would scale linearly with queue depth. The fix: a proper job queue (Celery / ARQ) feeding multiple inference workers, or a batched inference server (vLLM / TGI) that can process multiple prompts simultaneously.

### Stateful vs stateless — what are the tradeoffs?
This system is stateful: the FAISS index lives on disk and must be present for queries to work. The benefit is fast startup and no re-embedding cost. The tradeoff is that horizontal scaling is harder — each new replica needs access to the same index file (shared NFS, S3, or a managed vector DB). A stateless design (no local index, query a remote vector DB like Pinecone or Weaviate on every request) is easier to scale but adds network latency on every retrieval and introduces an external dependency.

### How is the LLM abstracted?
All LLM calls pass through a single `generate(prompt: str) -> str` function in `services/llm_service.py`. The query service calls `generate()` — it has no knowledge of Ollama, HTTP, or any backend detail. Swapping to OpenAI requires replacing one function:

```python
# Current: Ollama (local, free)
response = httpx.post("http://localhost:11434/api/generate", json={...})

# Swap to OpenAI — change only this function:
client = openai.OpenAI()
response = client.chat.completions.create(model="gpt-4o", messages=[...])
```

Zero changes to query_service.py, routes.py, or any other module.

---

## Running the Evaluations

```bash
# Retrieval benchmark (standalone — no server needed)
python -m tests.eval_retrieval

# Latency simulation (requires running server)
uvicorn main:app --host 0.0.0.0 --port 8000 &
python -m tests.eval_latency
```

Results are saved to `tests/results/`.

---

## Configuration

All parameters live in `config.yaml` — nothing is hardcoded in the codebase:

```yaml
chunking:
  default_strategy: "sentence"
  fixed:
    chunk_size: 256
    overlap: 32
embedding:
  model_name: "all-MiniLM-L6-v2"
  batch_size: 32
  device: "mps"
llm:
  provider: "ollama"
  model_name: "mistral"
  temperature: 0.1
```

---

## What This Demonstrates

- **RAG internals without a framework** — every component (chunking, embedding, vector search, prompt construction, LLM call) is implemented and visible, showing understanding of what LangChain abstracts away
- **System design thinking** — async ingestion vs sync querying, single responsibility at the service level, stateful/stateless tradeoffs documented
- **Production awareness** — observability endpoint, embedding cache, config-driven parameters, LLM abstraction layer with a documented swap path
- **Empirical evaluation** — retrieval quality benchmarked across strategies with real metrics, latency profiled and bottleneck identified with data