"""
FastAPI application entrypoint.

Startup sequence:
    1. Load config
    2. Attempt to load FAISS index from disk (non-fatal if none exists yet)
    3. Register routes

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import time

from fastapi import FastAPI

from core.config_loader import load_config
from core.faiss_store import load_index
from api.routes import router

_start_time = time.time()


def get_uptime() -> int:
    return int(time.time() - _start_time)


app = FastAPI(
    title="Document Q&A — RAG from Scratch",
    description="RAG pipeline built without LangChain: FAISS + Sentence-Transformers + Ollama.",
    version="1.0.0",
)

app.include_router(router)


@app.on_event("startup")
async def startup():
    cfg = load_config()
    print("\n[Startup] RAG server initialising...")
    print(f"[Startup] LLM backend : {cfg['llm']['provider']} / {cfg['llm']['model_name']}")
    print(f"[Startup] Embed model  : {cfg['embedding']['model_name']}")

    loaded = load_index()
    if loaded:
        print("[Startup] FAISS index loaded from disk — ready to query.")
    else:
        print("[Startup] No index found — send POST /index to begin ingestion.")

    print("[Startup] Server ready.\n")